import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evalkit import Evaluator
from evalkit.chexprompt import CheXpromptScorer, parse_rating_text


RATING = """Number of clinically significant errors by type: ((A, 1), (B, 2), (C, 0), (D, 0), (E, 0), (F, 0))
Number of clinically insignificant errors by type: [(A, 0), (B, 0), (C, 0), (D, 1), (E, 0), (F, 0)]"""


class Handler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        assert self.path == "/v1/chat/completions"
        assert self.headers["Authorization"] == "Bearer secret"
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["model"] == "judge-model"
        assert payload["messages"][0]["role"] == "system"
        body = json.dumps({"choices": [{"message": {"content": RATING}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def main():
    significant, insignificant = parse_rating_text(RATING)
    assert sum(significant.values()) == 3
    assert sum(insignificant.values()) == 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        # SQLite WAL cleanup can briefly lag on network filesystems.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            scorer = CheXpromptScorer(
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key="secret",
                model="judge-model",
                cache_path=os.path.join(tmp, "cache.sqlite3"),
                max_workers=2,
            )
            result = scorer.score_single("reference", "candidate")
            assert result["chexprompt_errors"] == 4.0
            assert result["chexprompt_reward"] == 0.2
            # Cached: the second score must not hit the HTTP server.
            assert scorer.score_single("reference", "candidate") == result
            assert Handler.calls == 1
            batch = scorer.score_batch(["r1", "r2"], ["h1", "h2"])
            assert len(batch) == 2
            os.environ["EVALKIT_CHEXPROMPT_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
            os.environ["EVALKIT_CHEXPROMPT_API_KEY"] = "secret"
            os.environ["EVALKIT_CHEXPROMPT_MODEL"] = "judge-model"
            integrated = Evaluator("chexprompt").score_single("ref", "hyp")
            assert integrated["chexprompt_errors"] == 4.0
            assert integrated["chexprompt_reward"] == 0.2
    finally:
        server.shutdown()
        server.server_close()
    print("CheXprompt tests passed")


if __name__ == "__main__":
    main()
