"""Seasonal IsolationForest models — one per season."""

import datetime
import logging
import os
import pickle

import pandas as pd

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
# Import feature columns from main module at runtime to stay in sync
def _get_feature_cols():
    try:
        from habitus.main import FEATURE_COLS as _FC
        return _FC
    except ImportError:
        return [
            "hour_of_day", "day_of_week", "is_weekend", "month",
            "total_power_w", "avg_temp_c", "sensor_changes",
            "lights_on", "motion_events", "presence_count",
            "people_home_pct", "media_active", "door_events",
            "outdoor_temp_c", "activity_diversity", "grid_kwh_w",
            "water_l_per_h", "water_leak", "gas_m3_per_h",
        ]

SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


def current_season():
    m = datetime.datetime.now().month
    for s, months in SEASONS.items():
        if m in months:
            return s
    return "winter"


def train_seasonal_models(features: pd.DataFrame):
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    saved = []
    for season, months in SEASONS.items():
        subset = features[features["month"].isin(months)]
        if len(subset) < 48:  # ~2 days minimum per season
            log.info(f"Season {season}: only {len(subset)}h data — skipping")
            continue
        X = subset[_get_feature_cols()].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        model.fit(Xs)
        mpath = os.path.join(DATA_DIR, f"model_{season}.pkl")
        spath = os.path.join(DATA_DIR, f"scaler_{season}.pkl")
        meta_path = os.path.join(DATA_DIR, f"meta_{season}.json")
        
        # Only overwrite if we have MORE data than existing model (extend, don't regress)
        existing_hours = 0
        try:
            import json as _json
            with open(meta_path) as _mf:
                existing_hours = _json.load(_mf).get("hours", 0)
        except Exception:
            pass
        
        if len(subset) >= existing_hours:
            with open(mpath, "wb") as f:
                pickle.dump(model, f)
            with open(spath, "wb") as f:
                pickle.dump(scaler, f)
            import json as _json
            with open(meta_path, "w") as _mf:
                _json.dump({"hours": len(subset), "days": len(subset) // 24}, _mf)
            saved.append(season)
            log.info(f"Season {season}: trained on {len(subset)}h ({len(subset)//24}d)")
        else:
            log.info(f"Season {season}: keeping existing model ({existing_hours}h > {len(subset)}h new)")
            saved.append(season)
    return saved


def score_with_best_model(X_raw):
    """Score using seasonal model if available, else fall back to main model."""
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
        Xs = scaler.transform(X_raw[:, :len(scaler.mean_)])
        raw = model.score_samples(Xs)[0]
        score = int(max(0, min(100, (-raw + 0.5) * 100)))
        return score, used
    except Exception as e:
        log.error(f"Scoring error: {e}")
        return 0, "error"


def seasonal_status():
    status = {}
    for season in SEASONS:
        mpath = os.path.join(DATA_DIR, f"model_{season}.pkl")
        status[season] = os.path.exists(mpath)
    return status
