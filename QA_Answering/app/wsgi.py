"""WSGI entrypoint.

Development:
    python app/wsgi.py                       # http://127.0.0.1:5000
    flask --app app.wsgi run                 # if app/ is on PYTHONPATH

Production:
    gunicorn --chdir app wsgi:app -b 0.0.0.0:5000 -w 2
"""

from __future__ import annotations

import os
import sys

# Make both the qa_system library (src/) and the qa_web package importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for path in (os.path.join(_ROOT, "src"), _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from qa_web import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("QA_HOST", "127.0.0.1"),
        port=int(os.environ.get("QA_PORT", "5000")),
        debug=os.environ.get("QA_DEBUG", "1") == "1",
    )
