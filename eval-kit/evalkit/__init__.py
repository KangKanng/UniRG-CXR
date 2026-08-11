"""eval-kit: self-contained NLG + radiology-report metrics.

Pure-Python NLG metrics (BLEU, CIDEr, ROUGE-L) ported from R2GenGPT/evalcap.
METEOR is intentionally omitted (requires a Java jar; see README).

Model metrics (chexbert = F1 CheXbert + SembScore; bertscore = BERTScore
P/R/F1; f1radgraph = F1 RadGraph; ratescore = RaTEScore) are loaded lazily
from local weights. CheXprompt uses a configured OpenAI-compatible endpoint.
"""
from .evaluator import Evaluator, METRIC_NAMES, get_metric
from .bleu import Bleu, BleuScorer
from .cider import Cider, CiderScorer
from .rouge import Rouge
from .chexprompt import CheXpromptScorer

__all__ = [
    "Evaluator",
    "METRIC_NAMES",
    "get_metric",
    "Bleu",
    "BleuScorer",
    "Cider",
    "CiderScorer",
    "Rouge",
    "CheXpromptScorer",
]

__version__ = "0.2.0"
