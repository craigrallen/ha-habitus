"""Progressive training — expands history window incrementally to avoid OOM.

Strategy:
  1. Train on last 30 days immediately → working model + score fast
  2. Background thread expands: 60d → 90d → 180d → 365d
  3. Each window retrains and publishes an updated score
  4. UI shows "Trained on 30d — expanding to 60d…"

This keeps peak RAM to ~130MB per pass (923 sensors × 30d × 24h rows).
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("habitus")

# Days windows to train through in order
WINDOWS = [30, 60, 90, 180, 365]

_thread: threading.Thread | None = None
_current_window: int = 0
_lock = threading.Lock()


def current_window() -> int:
    return _current_window


def is_expanding() -> bool:
    return _thread is not None and _thread.is_alive()


def start_progressive(max_days: int = 365) -> None:
    """Kick off the progressive training loop in a background thread."""
    global _thread
    with _lock:
        if is_expanding():
            log.info("Progressive training already running")
            return
        windows = [w for w in WINDOWS if w <= max_days]
        if not windows:
            windows = [max_days]
        if max_days not in windows:
            windows.append(max_days)
        _thread = threading.Thread(
            target=_loop,
            args=(windows,),
            daemon=True,
            name="habitus-progressive",
        )
        _thread.start()
        log.info("Progressive training started — windows: %s days", windows)


def _loop(windows: list[int]) -> None:
    global _current_window
    import asyncio

    from habitus.main import run  # noqa: PLC0415

    for days in windows:
        _current_window = days
        log.info("Progressive: training on last %d days", days)
        try:
            asyncio.run(run(days_history=days, mode="full"))
        except Exception:
            log.exception("Progressive training failed at %d days", days)
            break
        # Brief pause between windows so the device breathes
        time.sleep(10)

    _current_window = 0
    log.info("Progressive training complete")
