"""Extractive Question Answering with ranked, confidence-scored answers.

Modules
-------
config          : single source of truth for hyper-parameters and paths
data            : dataset loading (HF hub or local SQuAD-format JSON)
preprocessing   : tokenisation, sliding window, char-span -> token-span alignment
modeling        : model / tokenizer construction, save and reload
baseline        : out-of-the-box Hugging Face `question-answering` pipeline
train           : native PyTorch fine-tuning loop
postprocessing  : span extraction, confidence scoring, cross-passage ranking
metrics         : Exact Match / F1 plus Hit@k, Recall@k, MRR and calibration
evaluation      : batched inference + full metric report
inference       : RankedQAPipeline - the object the Flask app talks to
"""

from .config import QAConfig

__all__ = ["QAConfig", "RankedQAPipeline"]
__version__ = "2.0.0"


def __getattr__(name):
    # Lazy, so `import qa_system` does not pull in torch.
    if name == "RankedQAPipeline":
        from .inference import RankedQAPipeline
        return RankedQAPipeline
    raise AttributeError(name)
