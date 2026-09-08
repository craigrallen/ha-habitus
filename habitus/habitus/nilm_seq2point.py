"""NILM Seq2Point Disaggregation — appliance-level power decomposition.

Uses lightweight 1D-CNN seq2point models to disaggregate whole-house
power into individual appliance consumption. Models are small enough
to run on Raspberry Pi / ARM devices.

Architecture per appliance:
  Input:  599-sample window of aggregate power (1 Hz, ~10 min)
  Output: single midpoint power estimate for target appliance

Reference: Zhang et al., "Sequence-to-Point Learning with Neural
Networks for Non-Intrusive Load Monitoring" (2018).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from .utils import atomic_write

log = logging.getLogger("habitus")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
NILM_RESULTS_PATH = os.path.join(DATA_DIR, "nilm_seq2point.json")
MODEL_DIR = os.path.join(DATA_DIR, "nilm_models")

# Seq2point window size (samples)
WINDOW_SIZE = 599
MID_POINT = WINDOW_SIZE // 2

# Default appliance catalogue with typical power ranges (watts)
APPLIANCE_CATALOGUE: dict[str, dict[str, Any]] = {
    "kettle": {"min_w": 1800, "max_w": 3100, "typical_duration_min": 3},
    "microwave": {"min_w": 600, "max_w": 1500, "typical_duration_min": 5},
    "fridge": {"min_w": 50, "max_w": 300, "typical_duration_min": 30},
    "dishwasher": {"min_w": 200, "max_w": 2500, "typical_duration_min": 90},
    "washing_machine": {"min_w": 100, "max_w": 2500, "typical_duration_min": 60},
    "oven": {"min_w": 1000, "max_w": 3500, "typical_duration_min": 45},
    "dryer": {"min_w": 1500, "max_w": 3000, "typical_duration_min": 60},
    "heat_pump": {"min_w": 300, "max_w": 2500, "typical_duration_min": 120},
    "water_heater": {"min_w": 1000, "max_w": 4500, "typical_duration_min": 30},
    "ev_charger": {"min_w": 1400, "max_w": 11000, "typical_duration_min": 240},
}


class Seq2PointModel:
    """Lightweight 1D-CNN seq2point model for a single appliance.

    When no pre-trained model file is available, falls back to a simple
    power-band detector using the appliance's known wattage range.
    """

    def __init__(self, appliance: str, model_path: str | None = None) -> None:
        self.appliance = appliance
        self.model_path = model_path
        self._model: Any = None
        self._use_fallback = True

        catalogue = APPLIANCE_CATALOGUE.get(appliance, {})
        self.min_w = catalogue.get("min_w", 100)
        self.max_w = catalogue.get("max_w", 3000)
        self.typical_duration = catalogue.get("typical_duration_min", 30)

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a pre-trained model (ONNX or TFLite)."""
        try:
            if path.endswith(".onnx"):
                import onnxruntime as ort  # type: ignore[import-untyped]

                self._model = ort.InferenceSession(path)
                self._use_fallback = False
                log.info("Loaded ONNX model for %s", self.appliance)
            elif path.endswith(".tflite"):
                import tflite_runtime.interpreter as tflite  # type: ignore[import-untyped]

                self._model = tflite.Interpreter(model_path=path)
                self._model.allocate_tensors()
                self._use_fallback = False
                log.info("Loaded TFLite model for %s", self.appliance)
        except ImportError:
            log.info(
                "No runtime for %s model — using power-band fallback",
                path.rsplit(".", 1)[-1],
            )
        except (OSError, ValueError) as e:
            log.warning("Failed to load model %s: %s", path, e)

    def predict(self, window: np.ndarray) -> float:
        """Predict the appliance's power consumption for the window midpoint.

        Args:
            window: 1D array of aggregate power samples (length WINDOW_SIZE).

        Returns:
            Estimated power consumption in watts for the target appliance.
        """
        if len(window) < WINDOW_SIZE:
            window = np.pad(window, (0, WINDOW_SIZE - len(window)), mode="edge")

        if not self._use_fallback and self._model is not None:
            return self._predict_neural(window)
        return self._predict_band(window)

    def _predict_neural(self, window: np.ndarray) -> float:
        """Run neural network inference."""
        x = window.astype(np.float32).reshape(1, WINDOW_SIZE, 1)
        try:
            if hasattr(self._model, "run"):
                # ONNX
                input_name = self._model.get_inputs()[0].name
                result = self._model.run(None, {input_name: x})
                return float(max(0, result[0][0][0]))
            else:
                # TFLite
                input_details = self._model.get_input_details()
                output_details = self._model.get_output_details()
                self._model.set_tensor(input_details[0]["index"], x)
                self._model.invoke()
                result = self._model.get_tensor(output_details[0]["index"])
                return float(max(0, result[0][0]))
        except (IndexError, ValueError, RuntimeError) as e:
            log.warning("Neural inference failed for %s: %s", self.appliance, e)
            return self._predict_band(window)

    def _predict_band(self, window: np.ndarray) -> float:
        """Fallback: detect if aggregate power is in this appliance's band.

        Uses a simple heuristic: if the midpoint power is within the
        appliance's known range, estimate a contribution proportional
        to the overlap.
        """
        mid_power = float(window[MID_POINT])
        if mid_power < self.min_w * 0.8:
            return 0.0

        # Check if there's a step change consistent with this appliance
        left_avg = float(np.mean(window[: MID_POINT - 30]))
        delta = mid_power - left_avg

        if self.min_w * 0.7 <= delta <= self.max_w * 1.3:
            return float(np.clip(delta, self.min_w, self.max_w))
        return 0.0


class NilmDisaggregator:
    """Manages seq2point models for all known appliances."""

    def __init__(self) -> None:
        self.models: dict[str, Seq2PointModel] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Load available models from the model directory."""
        for appliance in APPLIANCE_CATALOGUE:
            model_path = None
            for ext in (".onnx", ".tflite"):
                candidate = os.path.join(MODEL_DIR, f"{appliance}{ext}")
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            self.models[appliance] = Seq2PointModel(appliance, model_path)
        log.info(
            "NILM disaggregator loaded: %d appliances (%d with neural models)",
            len(self.models),
            sum(1 for m in self.models.values() if not m._use_fallback),
        )

    def disaggregate(self, power_series: np.ndarray) -> dict[str, float]:
        """Disaggregate a power time series into per-appliance estimates.

        Args:
            power_series: 1D array of aggregate power samples (watts).
                Should be at least WINDOW_SIZE samples long.

        Returns:
            Dict mapping appliance name to estimated power (watts) at
            the midpoint of the series.
        """
        if len(power_series) < WINDOW_SIZE:
            power_series = np.pad(power_series, (0, WINDOW_SIZE - len(power_series)), mode="edge")

        results: dict[str, float] = {}
        total_assigned = 0.0
        mid_power = float(power_series[MID_POINT])

        for name, model in self.models.items():
            estimate = model.predict(power_series)
            if estimate > 0 and total_assigned + estimate <= mid_power:
                results[name] = round(estimate, 1)
                total_assigned += estimate

        # Residual (unidentified)
        results["_residual"] = round(max(0, mid_power - total_assigned), 1)
        results["_total"] = round(mid_power, 1)
        return results

    def run_on_history(self, hourly_power: list[float]) -> dict[str, Any]:
        """Run disaggregation on hourly power history.

        Args:
            hourly_power: List of hourly average power values (watts).

        Returns:
            Summary dict with per-appliance daily averages, detection
            counts, and estimated energy breakdown.
        """
        if not hourly_power or len(hourly_power) < 24:
            return {"error": "Need at least 24 hours of power data"}

        series = np.array(hourly_power, dtype=np.float32)

        # Upsample hourly to ~1Hz equivalent windows (simplified)
        appliance_totals: dict[str, float] = {}
        detection_counts: dict[str, int] = {}
        n_windows = max(1, len(series) - WINDOW_SIZE + 1)

        # Sample every 60 points for efficiency
        step = max(1, n_windows // (len(hourly_power) * 2))
        for i in range(0, n_windows, step):
            window = series[i : i + WINDOW_SIZE]
            if len(window) < WINDOW_SIZE:
                break
            result = self.disaggregate(window)
            for name, watts in result.items():
                if name.startswith("_"):
                    continue
                if watts > 0:
                    appliance_totals[name] = appliance_totals.get(name, 0) + watts
                    detection_counts[name] = detection_counts.get(name, 0) + 1

        n_samples = max(1, n_windows // step)
        breakdown: list[dict[str, Any]] = []
        for name in sorted(appliance_totals, key=lambda k: appliance_totals[k], reverse=True):
            avg_w = appliance_totals[name] / n_samples
            count = detection_counts.get(name, 0)
            if avg_w > 5:  # Minimum 5W to report
                breakdown.append(
                    {
                        "appliance": name,
                        "avg_power_w": round(avg_w, 1),
                        "detections": count,
                        "est_daily_kwh": round(avg_w * 24 / 1000, 2),
                        "catalogue": APPLIANCE_CATALOGUE.get(name, {}),
                    }
                )

        report = {
            "breakdown": breakdown,
            "total_avg_w": round(float(np.mean(series)), 1),
            "identified_pct": round(
                sum(b["avg_power_w"] for b in breakdown) / max(1, float(np.mean(series))) * 100,
                1,
            ),
            "n_appliances_detected": len(breakdown),
        }

        atomic_write(NILM_RESULTS_PATH, report)
        log.info(
            "NILM: %d appliances identified (%.1f%% of total load)",
            len(breakdown),
            report["identified_pct"],
        )
        return report
