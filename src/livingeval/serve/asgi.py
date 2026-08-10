"""ASGI entry point: `uvicorn livingeval.serve.asgi:app`.

Needed because `uvicorn --workers` and `gunicorn` require an import string rather than a
constructed app object — each worker process imports this module and builds its own.

Configuration comes entirely from the environment, which is what a container gives you.
Each worker holds its own fitted scorer, fitted from the same store, so they agree
without sharing memory. See `AppState` for why that is a real limitation rather than a
detail.
"""

from __future__ import annotations

from livingeval.config import Settings
from livingeval.serve.app import AppState, create_app
from livingeval.serve.middleware import configure_logging

__all__ = ["app", "settings", "state"]

settings = Settings.load()
configure_logging(settings.log_level, settings.log_json)
state = AppState(settings=settings)
app = create_app(state)
