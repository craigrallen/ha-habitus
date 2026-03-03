"""Seasonal IsolationForest models — one per season."""

import datetime
import json
import logging
import os
import pickle
from typing import Any

import pandas as pd

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
HEMISPHERE = os.environ.get("HEMISPHERE", "north")


def _get_feature_cols() -> list[str]:
    """Return the canonical feature-column list, importing from main at runtime.

    Falls back to a hardcoded list if the import fails (e.g. during testing).

    Returns:
        List of feature column names used for model training and scoring.
    """
    try:
        from habitus.main import FEATURE_COLS as _FC

        return list(_FC)
    except ImportError:
        return [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "month",
            "total_power_w",
            "avg_temp_c",
            "sensor_changes",
            "lights_on",
            "motion_events",
            "presence_count",
            "people_home_pct",
            "media_active",
            "door_events",
            "outdoor_temp_c",
            "activity_diversity",
            "grid_kwh_w",
            "water_l_per_h",
            "water_leak",
            "gas_m3_per_h",
        ]


# Northern-Hemisphere month groupings
SEASONS: dict[str, list[int]] = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}

# Southern Hemisphere flips summer/winter and spring/autumn
SOUTHERN_SEASON_MAP: dict[str, str] = {
    "winter": "summer",
    "summer": "winter",
    "spring": "autumn",
    "autumn": "spring",
}


def current_season(hemisphere: str | None = None) -> str:
    """Return the current meteorological season, hemisphere-aware.

    Args:
        hemisphere: ``"north"`` or ``"south"``. Defaults to the
            ``HEMISPHERE`` environment variable (default ``"north"``).

    Returns:
        One of ``"winter"``, ``"spring"``, ``"summer"``, ``"autumn"``.
    """
    hemi = hemisphere or HEMISPHERE
    m = datetime.datetime.now().month
    north_season = "winter"
    for s, months in SEASONS.items():
        if m in months:
            north_season = s
            break
    if hemi == "south":
        return SOUTHERN_SEASON_MAP[north_season]
    return north_season


def train_seasonal_models(features: pd.DataFrame) -> list[str]:
    """Train one IsolationForest per season and persist to ``DATA_DIR``.

    Saves individual season pickles (``model_{season}.pkl``,
    ``scaler_{season}.pkl``) **and** a combined ``seasonal_models.pkl``
    bundle for all successfully trained seasons.

    Args:
        features: Feature ``DataFrame`` containing a ``"month"`` column
            and all columns listed in ``FEATURE_COLS``.

    Returns:
        List of season names for which a model was successfully trained.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feature_cols = _get_feature_cols()
    saved: list[str] = []
    bundle_models: dict[str, Any] = {}
    bundle_scalers: dict[str, Any] = {}

    for season, months in SEASONS.items():
        subset = features[features["month"].isin(months)]
        if len(subset) < 48:  # ~2 days minimum per season
            log.info(f"Season {season}: only {len(subset)}h data — skipping")
            continue
        mpath = os.path.join(DATA_DIR, f"model_{season}.pkl")
        spath = os.path.join(DATA_DIR, f"scaler_{season}.pkl")
        meta_path = os.path.join(DATA_DIR, f"meta_{season}.json")

        # Only overwrite if we have MORE data than existing model (extend, don't regress)
        existing_hours = 0
        try:
            with open(meta_path) as _mf:
                existing_hours = json.load(_mf).get("hours", 0)
        except Exception:
            pass

        if len(subset) >= existing_hours:
            cols = [c for c in feature_cols if c in subset.columns]
            X = subset[cols].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            model.fit(Xs)
            with open(mpath, "wb") as f:
                pickle.dump(model, f)
            with open(spath, "wb") as f:
                pickle.dump(scaler, f)
            with open(meta_path, "w") as _mf:
                json.dump({"hours": len(subset), "days": len(subset) // 24}, _mf)
            bundle_models[season] = model
            bundle_scalers[season] = scaler
            log.info(f"Season {season}: trained on {len(subset)}h ({len(subset) // 24}d)")
        else:
            log.info(
                f"Season {season}: keeping existing model "
                f"({existing_hours}h > {len(subset)}h new)"
            )
            try:
                with open(mpath, "rb") as f:
                    bundle_models[season] = pickle.load(f)
                with open(spath, "rb") as f:
                    bundle_scalers[season] = pickle.load(f)
            except Exception:
                pass
        saved.append(season)

    if bundle_models:
        _save_seasonal_bundle(bundle_models, bundle_scalers)

    return saved


def _save_seasonal_bundle(models: dict[str, Any], scalers: dict[str, Any]) -> None:
    """Persist all seasonal models and scalers as a single bundle file.

    Writes ``seasonal_models.pkl`` to ``DATA_DIR`` containing a dict with
    ``"models"`` and ``"scalers"`` keys.

    Args:
        models: Mapping of season name to fitted ``IsolationForest``.
        scalers: Mapping of season name to fitted ``StandardScaler``.
    """
    bundle: dict[str, Any] = {"models": models, "scalers": scalers}
    path = os.path.join(DATA_DIR, "seasonal_models.pkl")
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Seasonal bundle saved ({len(models)} seasons) → {path}")


def load_seasonal_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the combined ``seasonal_models.pkl`` bundle from ``DATA_DIR``.

    Returns:
        Tuple of ``(models, scalers)`` where each is a dict keyed by
        season name. Returns ``({}, {})`` if the bundle does not exist.
    """
    path = os.path.join(DATA_DIR, "seasonal_models.pkl")
    if not os.path.exists(path):
        return {}, {}
    with open(path, "rb") as f:
        bundle: dict[str, Any] = pickle.load(f)
    return bundle.get("models", {}), bundle.get("scalers", {})


def score_with_best_model(X_raw: Any) -> tuple[int, str]:
    """Score using the current-season model, falling back to the main model.

    Selects the model for the hemisphere-aware current season; if that model
    has not been trained yet, falls back to ``model.pkl`` / ``scaler.pkl``.

    Args:
        X_raw: Raw feature array of shape ``(1, n_features)``.

    Returns:
        Tuple of ``(anomaly_score, model_identifier)`` where
        ``anomaly_score`` is 0–100 and ``model_identifier`` is the season
        name, ``"main"``, or ``"error"``.
    """
    season = current_season()
    spath = os.path.join(DATA_DIR, f"scaler_{season}.pkl")
    mpath = os.path.join(DATA_DIR, f"model_{season}.pkl")
    fallback_s = os.path.join(DATA_DIR, "scaler.pkl")
    fallback_m = os.path.join(DATA_DIR, "model.pkl")
    try:
        if os.path.exists(mpath):
            with open(mpath, "rb") as f:
                model = pickle.load(f)
            with open(spath, "rb") as f:
                scaler = pickle.load(f)
            used = season
        else:
            with open(fallback_m, "rb") as f:
                model = pickle.load(f)
            with open(fallback_s, "rb") as f:
                scaler = pickle.load(f)
            used = "main"
        Xs = scaler.transform(X_raw[:, : len(scaler.mean_)])
        raw = model.score_samples(Xs)[0]
        score = int(max(0, min(100, (-raw + 0.5) * 100)))
        return score, used
    except Exception as e:
        log.error(f"Scoring error: {e}")
        return 0, "error"


def seasonal_status() -> dict[str, bool]:
    """Return presence status of each seasonal model file.

    Returns:
        Dict mapping each season name to ``True`` if its ``model_{season}.pkl``
        exists in ``DATA_DIR``, ``False`` otherwise.
    """
    status: dict[str, bool] = {}
    for season in SEASONS:
        mpath = os.path.join(DATA_DIR, f"model_{season}.pkl")
        status[season] = os.path.exists(mpath)
    return status
