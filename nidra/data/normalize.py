"""Feature normalization: RobustScaler fit on the TRAINING split only,
serialized alongside the model, never refit at serving or evaluation time.

This is called out repeatedly in the spec as the single most common source
of silent leakage in this class of project. The enforcement mechanism here
is structural: `FeatureScaler.fit` is only ever called from the training
pipeline on `train` rows, and `transform` refuses to run before `fit`/`load`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler

from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER, LOG1P_FEATURES, SCHEMA_VERSION


class FeatureScaler:
    def __init__(self, clip_min: float = -10.0, clip_max: float = 10.0):
        self.scaler = RobustScaler()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.fitted = False

    def _apply_log1p(self, X: np.ndarray) -> np.ndarray:
        X = X.astype("float64", copy=True)
        for name in LOG1P_FEATURES:
            idx = FEATURE_INDEX[name]
            X[..., idx] = np.log1p(np.clip(X[..., idx], a_min=0.0, a_max=None))
        return X

    def fit(self, X_train: np.ndarray) -> "FeatureScaler":
        """Fit on TRAINING data only. `X_train` must be [N, 45]."""
        if X_train.shape[-1] != len(FEATURE_ORDER):
            raise ValueError(f"expected {len(FEATURE_ORDER)} features, got {X_train.shape[-1]}")
        Xl = self._apply_log1p(X_train)
        flat = Xl.reshape(-1, Xl.shape[-1])
        self.scaler.fit(flat)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler.transform called before fit()/load() — never refit at serving time")
        if X.shape[-1] != len(FEATURE_ORDER):
            raise ValueError(f"expected {len(FEATURE_ORDER)} features, got {X.shape[-1]}")
        orig_shape = X.shape
        Xl = self._apply_log1p(X)
        flat = Xl.reshape(-1, Xl.shape[-1])
        Xs = self.scaler.transform(flat).reshape(orig_shape)
        return np.clip(Xs, self.clip_min, self.clip_max)

    def save(self, scaler_path: str | Path, metadata_path: str | Path, extra_metadata: dict | None = None) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted FeatureScaler")
        scaler_path, metadata_path = Path(scaler_path), Path(metadata_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        meta = {
            "feature_order": FEATURE_ORDER,
            "log1p_features": LOG1P_FEATURES,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "schema_version": SCHEMA_VERSION,
            "method": "robust_scaler",
        }
        if extra_metadata:
            meta.update(extra_metadata)
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, scaler_path: str | Path, metadata_path: str | Path) -> "FeatureScaler":
        with open(metadata_path) as f:
            meta = json.load(f)
        if meta["feature_order"] != FEATURE_ORDER:
            raise ValueError(
                "Serialized scaler's feature order does not match the current "
                "FEATURE_ORDER — schema drift. Refusing to load."
            )
        obj = cls(clip_min=meta["clip_min"], clip_max=meta["clip_max"])
        obj.scaler = joblib.load(scaler_path)
        obj.fitted = True
        return obj
