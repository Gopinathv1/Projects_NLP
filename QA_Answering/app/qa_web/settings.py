"""Environment-driven Flask configuration.

Kept separate from the app factory so settings can be swapped per environment
(dev / prod / test) without touching wiring code. Values come from environment
variables with sensible defaults, which is the twelve-factor convention.
"""

from __future__ import annotations

import os


class BaseConfig:
    # Where the fine-tuned checkpoint lives. If absent, the service falls back
    # to FALLBACK_MODEL so the API is never dead.
    MODEL_DIR: str = os.environ.get("QA_MODEL_DIR", "artifacts/qa-model")
    FALLBACK_MODEL: str = os.environ.get("QA_FALLBACK_MODEL",
                                         "distilbert-base-cased-distilled-squad")
    BASELINE_MODEL: str = os.environ.get("QA_BASELINE_MODEL",
                                         "distilbert-base-cased-distilled-squad")

    # Request guard rails - reject obviously abusive payloads before inference.
    MAX_PASSAGES: int = int(os.environ.get("QA_MAX_PASSAGES", "20"))
    MAX_PASSAGE_CHARS: int = int(os.environ.get("QA_MAX_PASSAGE_CHARS", "50000"))
    MAX_QUESTION_CHARS: int = int(os.environ.get("QA_MAX_QUESTION_CHARS", "1000"))

    JSON_SORT_KEYS = False
    LOG_LEVEL: str = os.environ.get("QA_LOG_LEVEL", "INFO")

    # Load the model when the app boots (True) or on the first request (False).
    EAGER_LOAD: bool = os.environ.get("QA_EAGER_LOAD", "0") == "1"


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False


class TestConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    EAGER_LOAD = False


_CONFIGS = {"dev": DevConfig, "development": DevConfig,
            "prod": ProdConfig, "production": ProdConfig,
            "test": TestConfig, "testing": TestConfig}


def resolve_config(name: str | None = None):
    """Pick a config class by name or the QA_ENV environment variable."""
    key = (name or os.environ.get("QA_ENV", "dev")).lower()
    return _CONFIGS.get(key, DevConfig)
