"""HTML routes - just renders the single-page interface."""

from __future__ import annotations

from flask import Blueprint, render_template

from ..samples import SAMPLES

web_bp = Blueprint("web", __name__)


@web_bp.get("/")
def index():
    return render_template("index.html", samples=SAMPLES)
