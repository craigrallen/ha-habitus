"""Multi-dwelling support — separate behavioral profiles per location.

Allows Habitus to manage multiple dwellings (e.g. house, boat, cabin)
each with independent models, baselines, and anomaly data.  The active
dwelling is determined by which person entities report "home".
"""

import json
import logging
import os
import uuid
from typing import Any

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
DWELLING_PATH = os.path.join(DATA_DIR, "dwellings.json")


def _load_dwellings() -> dict[str, Any]:
    """Load the dwellings registry from disk.

    Returns:
        Mapping of dwelling name to config dict.  Empty dict if the
        file does not exist or is corrupt.
    """
    if not os.path.exists(DWELLING_PATH):
        return {}
    try:
        with open(DWELLING_PATH) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load dwellings.json: %s", exc)
    return {}


def _save_dwellings(dwellings: dict[str, Any]) -> None:
    """Persist the dwellings registry atomically.

    Args:
        dwellings: Full dwellings mapping to write.
    """
    atomic_write(DWELLING_PATH, dwellings)


def _dwelling_dir(name: str) -> str:
    """Return the data subdirectory for a named dwelling.

    Args:
        name: Dwelling name (used as directory slug).

    Returns:
        Absolute path under DATA_DIR/dwellings/{name}/.
    """
    safe_name = name.lower().replace(" ", "_")
    return os.path.join(DATA_DIR, "dwellings", safe_name)


def register_dwelling(name: str, person_entities: list[str]) -> dict:
    """Register a new dwelling with its associated person entities.

    Creates a subdirectory under DATA_DIR for the dwelling's models and
    baselines.  If the dwelling already exists, its person entities are
    updated in place.

    Args:
        name: Human-friendly dwelling name (e.g. "Cabin").
        person_entities: List of HA person entity IDs whose "home" state
            indicates presence at this dwelling.

    Returns:
        The dwelling config dict that was persisted.
    """
    dwellings = _load_dwellings()
    dwell_dir = _dwelling_dir(name)
    os.makedirs(dwell_dir, exist_ok=True)

    config: dict[str, Any] = dwellings.get(name, {})
    config.update(
        {
            "name": name,
            "person_entities": person_entities,
            "data_dir": dwell_dir,
            "id": config.get("id", uuid.uuid4().hex[:8]),
        }
    )
    dwellings[name] = config
    _save_dwellings(dwellings)
    log.info("Registered dwelling '%s' with %d person entities", name, len(person_entities))
    return config


def detect_active_dwelling(ha_states: list[dict]) -> str:
    """Determine which dwelling is currently active based on person entities.

    Iterates registered dwellings and checks which person entities report
    a state of ``"home"``.  The dwelling with the most "home" person
    entities wins.  Falls back to ``"default"`` if no match.

    Args:
        ha_states: Raw HA ``/api/states`` response (list of entity dicts).

    Returns:
        Name of the active dwelling, or ``"default"``.
    """
    dwellings = _load_dwellings()
    if not dwellings:
        return "default"

    state_map: dict[str, str] = {}
    for s in ha_states:
        eid = s.get("entity_id", "")
        if eid.startswith("person."):
            state_map[eid] = s.get("state", "unknown")

    best_name = "default"
    best_count = 0

    for name, cfg in dwellings.items():
        home_count = sum(1 for pe in cfg.get("person_entities", []) if state_map.get(pe) == "home")
        if home_count > best_count:
            best_count = home_count
            best_name = name

    log.debug("Active dwelling detected: %s (%d home entities)", best_name, best_count)
    return best_name


def get_dwelling_config(dwelling_name: str) -> dict:
    """Return model and baseline paths for a specific dwelling.

    Args:
        dwelling_name: Registered dwelling name.

    Returns:
        Dict with ``data_dir``, ``model_path``, ``scaler_path``,
        ``baseline_path``, and ``entity_baselines_path``.

    Raises:
        KeyError: If the dwelling is not registered.
    """
    dwellings = _load_dwellings()
    if dwelling_name not in dwellings:
        raise KeyError(f"Dwelling '{dwelling_name}' is not registered")

    dwell_dir = dwellings[dwelling_name]["data_dir"]
    return {
        "name": dwelling_name,
        "data_dir": dwell_dir,
        "model_path": os.path.join(dwell_dir, "model.pkl"),
        "scaler_path": os.path.join(dwell_dir, "scaler.pkl"),
        "baseline_path": os.path.join(dwell_dir, "baseline.json"),
        "entity_baselines_path": os.path.join(dwell_dir, "entity_baselines.json"),
    }


def switch_dwelling(dwelling_name: str) -> dict:
    """Switch the active profile to a different dwelling.

    Updates the ``HABITUS_DWELLING`` environment variable and returns
    the dwelling config so callers can load the correct models.

    Args:
        dwelling_name: Registered dwelling name to activate.

    Returns:
        The dwelling config dict (same as :func:`get_dwelling_config`).

    Raises:
        KeyError: If the dwelling is not registered.
    """
    config = get_dwelling_config(dwelling_name)
    os.environ["HABITUS_DWELLING"] = dwelling_name
    log.info("Switched active dwelling to '%s' (data_dir=%s)", dwelling_name, config["data_dir"])
    return config


def list_dwellings() -> list[dict]:
    """List all registered dwellings.

    Returns:
        List of dwelling config dicts, one per registered location.
    """
    dwellings = _load_dwellings()
    return list(dwellings.values())
