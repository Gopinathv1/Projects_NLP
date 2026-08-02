"""Modular Flask application for ranked extractive QA (Requirement 4.6).

Package layout
--------------
    qa_web/
      __init__.py            <- create_app() application factory (this file)
      settings.py            <- environment-driven config classes
      errors.py              <- ApiError + centralised error handlers
      samples.py             <- demo passages shared by UI and scripts
      logging_utils.py       <- one place to configure logging
      blueprints/
        web.py               <- HTML route (the single-page UI)
        api.py               <- JSON API: /api/health, /api/info, /api/answer
      services/
        qa_service.py        <- model lifecycle + engine dispatch (the only torch touchpoint)
        validation.py        <- request parsing / guard rails
      templates/, static/    <- front end

The factory pattern keeps construction declarative and lets tests build an app
with a stub service and no model download. Routes depend on the service through
`app.extensions['qa_service']`, never by importing the model directly.
"""

from __future__ import annotations

from flask import Flask

from .blueprints import api_bp, web_bp
from .errors import register_error_handlers
from .logging_utils import configure_logging
from .services import QAService
from .settings import resolve_config

__all__ = ["create_app"]


def create_app(config_name: str | None = None, qa_service: QAService | None = None) -> Flask:
    """Build and wire the application.

    Parameters
    ----------
    config_name : "dev" | "prod" | "test" (falls back to the QA_ENV env var).
    qa_service  : inject a ready service (tests pass a stub); otherwise one is
                  built from config.
    """
    config = resolve_config(config_name)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config)
    app.config_object = config          # keep the class handy for validators
    app.json.sort_keys = getattr(config, "JSON_SORT_KEYS", False)

    configure_logging(app)

    # --- service layer -----------------------------------------------------
    service = qa_service or QAService.from_config(config)
    app.extensions["qa_service"] = service
    if getattr(config, "EAGER_LOAD", False):
        app.logger.info("EAGER_LOAD set - warming up the model at boot")
        service.warmup()

    # --- routing + errors --------------------------------------------------
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    register_error_handlers(app)

    app.logger.info("QA web app ready (env=%s, model_dir=%s)",
                    config.__name__, config.MODEL_DIR)
    return app
