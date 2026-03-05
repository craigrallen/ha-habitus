"""Sequential Pattern Mining using PrefixSpan.

Discovers ordered routines: "hallway motion → kitchen lights → kettle → coffee machine"
Unlike co-occurrence (scenes), this captures the FLOW of behaviour.
"""

import datetime
import json
import logging
import os
import sqlite3
from collections import defaultdict
from typing import Any

from .ha_db import resolve_ha_db_path

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SEQUENCES_PATH = os.path.join(DATA_DIR, "sequences.json")

ACTION_DOMAINS = ("light", "switch", "media_player", "climate", "fan", "cover",
                  "binary_sensor", "input_boolean")
# Max gap between events in a sequence (minutes)
MAX_GAP_MIN = 10
# Min gap to separate sequences (minutes)
MIN_SEQ_GAP_MIN = 30
# Minimum support (how many times a sequence must appear)
MIN_SUPPORT = 3


def _load_event_streams(entity_to_area: dict[str, str], days: int = 30) -> list[list[str]]:
    """Load state changes as sequences of symbolic events.

    Groups events into sessions (separated by MIN_SEQ_GAP_MIN gaps).
    Each event = "room:entity:state" (e.g. "Kitchen:light.ceiling:on").
    """
    db_path = resolve_ha_db_path()
    if not db_path:
        return []

    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    like_clauses = " OR ".join(f"sm.entity_id LIKE '{d}.%'" for d in ACTION_DOMAINS)

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(f"""
            SELECT sm.entity_id, s.state, s.last_changed_ts
            FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE s.last_changed_ts > ?
            AND ({like_clauses})
            AND s.state NOT IN ('unavailable', 'unknown', '')
            ORDER BY s.last_changed_ts
        """, (cutoff_ts,)).fetchall()
        conn.close()
    except Exception as e:
        log.warning("sequence_miner: DB query failed: %s", e)
        return []

    if not rows:
        return []

    # Convert to symbolic events and split into sessions
    sessions: list[list[str]] = []
    current_session: list[str] = []
    last_ts = 0.0

    for eid, state, ts in rows:
        if state in ("on", "off", "heat", "cool", "auto", "playing", "paused", "idle",
                      "home", "not_home", "open", "closed"):
            room = entity_to_area.get(eid, "unknown")
            symbol = f"{room}:{eid.split('.')[-1]}:{state}"

            if last_ts > 0 and (ts - last_ts) > MIN_SEQ_GAP_MIN * 60:
                if len(current_session) >= 2:
                    sessions.append(current_session)
                current_session = []

            # Skip if same as last event (dedup)
            if not current_session or current_session[-1] != symbol:
                # Check gap within sequence
                if last_ts > 0 and (ts - last_ts) <= MAX_GAP_MIN * 60:
                    current_session.append(symbol)
                elif not current_session:
                    current_session.append(symbol)
                else:
                    if len(current_session) >= 2:
                        sessions.append(current_session)
                    current_session = [symbol]
            last_ts = ts

    if len(current_session) >= 2:
        sessions.append(current_session)

    log.info("sequence_miner: %d sessions from %d raw events", len(sessions), len(rows))
    return sessions


def mine_sequences(entity_to_area: dict[str, str], days: int = 30) -> dict[str, Any]:
    """Run PrefixSpan sequential pattern mining."""
    sessions = _load_event_streams(entity_to_area, days=days)
    if len(sessions) < MIN_SUPPORT:
        return {"sequences": [], "total_sessions": len(sessions)}

    try:
        from prefixspan import PrefixSpan
    except ImportError:
        log.warning("prefixspan not installed, skipping sequence mining")
        return {"sequences": [], "error": "prefixspan not installed"}

    # Cap session length to avoid explosion
    capped = [s[:30] for s in sessions]

    ps = PrefixSpan(capped)
    ps.minlen = 2
    ps.maxlen = 8

    # Find frequent sequences
    frequent = ps.frequent(MIN_SUPPORT)

    # Convert to readable format
    results = []
    for support, pattern in sorted(frequent, key=lambda x: -x[0])[:50]:
        # Extract rooms and entities
        rooms = set()
        entities = []
        for symbol in pattern:
            parts = symbol.split(":")
            if len(parts) >= 3:
                rooms.add(parts[0])
                entities.append({"entity": parts[1], "state": parts[2], "room": parts[0]})

        # Classify the sequence
        if len(pattern) >= 3:
            category = "routine"
        elif any("motion" in s or "pir" in s for s in pattern):
            category = "trigger_chain"
        else:
            category = "pair"

        description = " → ".join(
            f"{e['entity'].replace('_',' ').title()} ({e['state']})"
            for e in entities
        )

        results.append({
            "pattern": pattern,
            "support": support,
            "frequency_pct": round(support / len(sessions) * 100, 1),
            "length": len(pattern),
            "rooms": sorted(rooms),
            "entities": entities,
            "category": category,
            "description": description,
        })

    result = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "total_sessions": len(sessions),
        "total_patterns": len(results),
        "sequences": results,
    }

    from .utils import atomic_write as _atomic_write  # noqa: PLC0415
    _atomic_write(SEQUENCES_PATH, result)

    log.info("sequence_miner: found %d frequent patterns from %d sessions", len(results), len(sessions))
    return result
