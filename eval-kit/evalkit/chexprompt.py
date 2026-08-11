"""CheXprompt scoring through an OpenAI-compatible chat-completions API.

The rubric and five in-context examples follow the official Microsoft
CheXprompt implementation.  This module intentionally uses only the Python
standard library so adding the remote metric cannot change a training
environment's torch/openai packages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Mapping, Sequence, Tuple


def _load_dotenv() -> None:
    """Load eval-kit/.env without adding a dependency on python-dotenv.

    Existing process environment variables take precedence. Set
    ``EVALKIT_ENV_FILE`` to use a file outside the repository.
    """
    configured = os.getenv("EVALKIT_ENV_FILE")
    repo_env = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    candidates = [configured] if configured else [repo_env, os.path.join(os.getcwd(), ".env")]
    path = next((p for p in candidates if p and os.path.isfile(p)), None)
    if path is None:
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


ERROR_TYPES = ("A", "B", "C", "D", "E", "F")
ERROR_TYPE_MAP = {
    "A": "false_positive_finding",
    "B": "omission_finding",
    "C": "incorrect_location",
    "D": "incorrect_severity",
    "E": "false_positive_comparison",
    "F": "omission_comparison",
}

SYSTEM_PROMPT = (
    "Instructions: You are an expert radiologist. Judge the diagnostic accuracy "
    "of generated radiology report findings based on a reference findings "
    "section. For each error type, count how many errors exist in the candidate "
    "report. Examples are provided for you. For clinically significant and "
    "clinically insignificant errors of 6 error types, count how many of each "
    "error type there are. Refer to the reference and candidate findings as "
    "needed to keep maximum accuracy in counting each error type. Finally, "
    "provide the error counts in the list format exactly as it is given to you."
)

EXAMPLES = '''##
Reference Findings: """Dilated distal esophagus as seen previously containing ingested food contents.  No signs of aspiration.  Please refer to prior CT torso for full descriptive details of esophageal abnormalities."""

Candidate Findings: """Dobbhoff terminates in the distal esophagus ."""

Errors: Number of clinically significant errors by type: ((A, 1), (B, 1), (C, 0), (D, 0), (E, 0), (F, 1))
Number of clinically insignificant errors by type: ((A, 0), (B, 0), (C, 0), (D, 0), (E, 0), (F, 0))
##
Reference Findings: """PA and lateral chest compared to ___ and ___:  Mild pulmonary edema is less severe today than it was on ___.  Small pleural effusions and moderate cardiomegaly are comparable.  There is no pneumonia.  Very small right upper lobe lung nodule may be present projected over the intersection of the right first anterior and fifth posterior ribs.  Findings were discussed by Dr. ___ with Dr. ___ by telephone at the time of this dictation."""

Candidate Findings: """1. Mildly improved pulmonary edema with increased cardiomegaly, now moderate. 2. Small right pleural effusion, better assessed on prior chest CTA, likely unchanged.  No effusion on the left. 3. No evidence of pneumonia."""

Errors: Number of clinically significant errors by type: ((A, 0), (B, 1), (C, 0), (D, 1), (E, 0), (F, 0))
Number of clinically insignificant errors by type: ((A, 0), (B, 1), (C, 0), (D, 0), (E, 0), (F, 0))
##
Reference Findings: """In comparison with the study of ___, the monitoring and support devices are unchanged.  Opacification at the right base is unchanged, again consistent with collapse of the middle and lower lobes.  The left lung remains clear."""

Candidate Findings: """In comparison with the study of ___ , the monitoring and support devices essentially unchanged . Continued low lung volumes without definite vascular congestion . The right base is clear on this study . Opacification at the left base is consistent with small effusion and atelectatic changes ."""

Errors: Number of clinically significant errors by type: ((A, 1), (B, 1), (C, 0), (D, 0), (E, 0), (F, 0))
Number of clinically insignificant errors by type: ((A, 1), (B, 0), (C, 0), (D, 0), (E, 0), (F, 0))
##
Reference Findings: """Heart size is enlarged but stable. There remains moderate pulmonary edema which is unchanged. There is an unchanged left retrocardiac opacity. There are likely small bilateral effusions. There are no pneumothoraces."""

Candidate Findings: """Heart size is upper limits of normal but stable.  There is persistent mild pulmonary edema. There is a left retrocardiac opacity, stable. There are no pneumothoraces."""

Errors: Number of clinically significant errors by type: ((A, 0), (B, 0), (C, 0), (D, 1), (E, 0), (F, 0))
Number of clinically insignificant errors by type: ((A, 0), (B, 0), (C, 0), (D, 0), (E, 0), (F, 0))
##
Reference Findings: """1.  Left retrocardiac opacification could be atelectasis or infection.  2.  Pulmonary vascular congestion without evidence of interstitial edema.  3.  Possible small left pleural effusion."""

Candidate Findings: """1.  Pulmonary vascular congestion without frank interstitial edema.  2.  Small bilateral pleural effusions.  3.  Subsegmental bilateral lower lobe atelectasis."""

Errors: Number of clinically significant errors by type: ((A, 0), (B, 0), (C, 0), (D, 0), (E, 0), (F, 0))
Number of clinically insignificant errors by type: ((A, 1), (B, 0), (C, 0), (D, 0), (E, 0), (F, 0))
##'''

USER_PROMPT = '''A clinically significant error is one that likely affects treatment, management, or outcomes. There are six error types:
A) False prediction of finding that is not present in the reference findings
B) Omission of finding that is present in the reference findings
C) Incorrect location/position of finding in the candidate findings compared to the reference findings
D) Incorrect severity of finding in the candidate findings compared to the reference findings
E) Mention of comparison that is not present in the reference findings
F) Omission of comparison describing a change from a previous study that is present in the reference findings

Desired output format:
Number of clinically significant errors by type: [(A, n_A), (B, n_B), (C, n_C), (D, n_D), (E, n_E), (F, n_F)]
Number of clinically insignificant errors by type: [(A, n_A), (B, n_B), (C, n_C), (D, n_D), (E, n_E), (F, n_F)]
##
{examples}
##
Reference Findings: """{reference}"""

Candidate Findings: """{candidate}"""

Errors:'''

_RATING_RE = re.compile(
    r"Number of clinically significant errors by type\s*:\s*(.*?)\s*"
    r"Number of clinically insignificant errors by type\s*:\s*(.*?)(?:\n\s*Explanation|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_PAIR_RE = re.compile(r"[\(\[]\s*([A-F])\s*,\s*(\d+)\s*[\)\]]", re.IGNORECASE)


def parse_rating_text(text: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Parse and strictly validate the two six-category CheXprompt ratings."""
    match = _RATING_RE.search(text or "")
    if not match:
        raise ValueError("CheXprompt response does not contain both rating lines")

    parsed: List[Dict[str, int]] = []
    for section in match.groups():
        raw = {letter.upper(): int(value) for letter, value in _PAIR_RE.findall(section)}
        if set(raw) != set(ERROR_TYPES):
            raise ValueError(f"CheXprompt response has invalid error categories: {sorted(raw)}")
        parsed.append({ERROR_TYPE_MAP[k]: raw[k] for k in ERROR_TYPES})
    return parsed[0], parsed[1]


class _SqliteCache:
    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._local = threading.local()
        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chexprompt_cache "
            "(cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL)"
        )
        conn.commit()

    def _connect(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            self._local.conn = conn
        return conn

    def get(self, key: str):
        row = self._connect().execute(
            "SELECT result_json FROM chexprompt_cache WHERE cache_key=?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value: Mapping[str, float]):
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO chexprompt_cache(cache_key, result_json) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
        conn.commit()


class CheXpromptScorer:
    """Official CheXprompt rubric with a configurable OpenAI-compatible API.

    Configuration may be passed explicitly or through:
    ``EVALKIT_CHEXPROMPT_BASE_URL``, ``EVALKIT_CHEXPROMPT_API_KEY``, and
    ``EVALKIT_CHEXPROMPT_MODEL``.  ``base_url`` can be either an API root such
    as ``https://host/v1`` or the full ``.../chat/completions`` endpoint.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        max_workers: int = 8,
        requests_per_minute: int = 0,
        cache_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_p: float = 0.9,
    ):
        _load_dotenv()
        base_url = base_url or os.getenv("EVALKIT_CHEXPROMPT_BASE_URL")
        api_key = api_key or os.getenv("EVALKIT_CHEXPROMPT_API_KEY")
        model = model or os.getenv("EVALKIT_CHEXPROMPT_MODEL")
        if not base_url:
            raise ValueError("set base_url or EVALKIT_CHEXPROMPT_BASE_URL")
        if not api_key:
            raise ValueError("set api_key or EVALKIT_CHEXPROMPT_API_KEY")
        if not model:
            raise ValueError("set model or EVALKIT_CHEXPROMPT_MODEL")
        root = base_url.rstrip("/")
        self.url = root if root.endswith("/chat/completions") else root + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_workers = max(1, max_workers)
        self.requests_per_minute = max(0, requests_per_minute)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0
        cache_path = cache_path or os.getenv("EVALKIT_CHEXPROMPT_CACHE")
        self.cache = _SqliteCache(cache_path) if cache_path else None

    @staticmethod
    def format_messages(reference: str, candidate: str) -> List[Dict[str, str]]:
        user = USER_PROMPT.format(
            examples=EXAMPLES, reference=reference, candidate=candidate
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _cache_key(self, reference: str, candidate: str) -> str:
        payload = json.dumps(
            {"version": 1, "model": self.model, "reference": reference, "candidate": candidate},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _throttle(self):
        if self.requests_per_minute <= 0:
            return
        interval = 60.0 / self.requests_per_minute
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + interval
        if wait:
            time.sleep(wait)

    def _request(self, reference: str, candidate: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": self.format_messages(reference, candidate),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if not isinstance(content, str):
                    raise ValueError("API response message.content is not text")
                return content
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"CheXprompt API HTTP {exc.code}: {detail}")
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    break
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                delay = 2 ** attempt
            if attempt < self.max_retries:
                logging.warning("CheXprompt request failed (attempt %d): %s", attempt + 1, last_error)
                time.sleep(delay)
        raise RuntimeError(f"CheXprompt API failed after {self.max_retries + 1} attempts") from last_error

    def score_single(self, reference: str, candidate: str) -> Dict[str, float]:
        if not isinstance(reference, str) or not isinstance(candidate, str):
            raise TypeError("reference and candidate must be strings")
        key = self._cache_key(reference, candidate)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        last_error = None
        for parse_attempt in range(self.max_retries + 1):
            try:
                significant, insignificant = parse_rating_text(
                    self._request(reference, candidate)
                )
                break
            except ValueError as exc:
                last_error = exc
                if parse_attempt == self.max_retries:
                    raise RuntimeError("CheXprompt returned an invalid rating") from last_error
                logging.warning("Invalid CheXprompt rating; regenerating: %s", exc)

        significant_total = sum(significant.values())
        insignificant_total = sum(insignificant.values())
        total = significant_total + insignificant_total
        result: Dict[str, float] = {
            "chexprompt_errors": float(total),
            "chexprompt_significant_errors": float(significant_total),
            "chexprompt_insignificant_errors": float(insignificant_total),
            "chexprompt_reward": 1.0 / (total + 1.0),
        }
        for name, value in significant.items():
            result[f"chexprompt_significant_{name}"] = float(value)
        for name, value in insignificant.items():
            result[f"chexprompt_insignificant_{name}"] = float(value)
        if self.cache:
            self.cache.put(key, result)
        return result

    def score_batch(self, refs: Sequence[str], hypos: Sequence[str]) -> List[Dict[str, float]]:
        if len(refs) != len(hypos):
            raise ValueError(f"refs/hypos length mismatch: {len(refs)} != {len(hypos)}")
        results: List[Dict[str, float] | None] = [None] * len(refs)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.score_single, ref, hypo): i
                for i, (ref, hypo) in enumerate(zip(refs, hypos))
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [item for item in results if item is not None]

    def compute(self, gts, res):
        keys = list(gts.keys())
        refs = [gts[k][0] for k in keys]
        hypos = [res[k][0] for k in keys]
        per_item = self.score_batch(refs, hypos)
        corpus = {
            key: sum(item[key] for item in per_item) / len(per_item)
            for key in per_item[0]
        } if per_item else {}
        return corpus, per_item
