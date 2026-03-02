"""Background training thread — runs the ML pipeline without blocking the web server."""

from __future__ import annotations

import asyncio
import logging
import threading

log = logging.getLogger("habitus")

_thread: threading.Thread | None = None
_lock = threading.Lock()


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def start(days: int, mode: str = "full") -> bool:
    """Spawn a background training thread.  Returns False if already running."""
    global _thread
    with _lock:
        if is_running():
            log.info("Training already in progress — ignoring start request")
            return False
        _thread = threading.Thread(
            target=_run_blocking,
            args=(days, mode),
            daemon=True,
            name="habitus-trainer",
        )
        _thread.start()
        log.info("Background training started (mode=%s, days=%d)", mode, days)
        return True


def _run_blocking(days: int, mode: str) -> None:
    """Execute the async training pipeline in a fresh event loop."""
    try:
        from habitus.main import run  # noqa: PLC0415

        asyncio.run(run(days_history=days, mode=mode))
    except Exception:
        log.exception("Background training failed")
