"""Root entry point for Hugging Face Spaces and Docker.

HF Spaces auto-discovers `app.py` at the repo root. The actual app lives in
`app/main.py`; this shim launches it on `0.0.0.0:7860` (or whatever
`GRADIO_SERVER_*` env vars say).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr

from app.main import _BANKY_CSS, _BANKY_THEME, build_app


if __name__ == "__main__":
    demo = build_app()
    # Explicit queue so streaming events + multiple clients don't block each
    # other (and tab switches don't freeze when a chat turn is in flight).
    demo.queue(default_concurrency_limit=4, max_size=16)
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        theme=_BANKY_THEME,
        css=_BANKY_CSS,
    )
