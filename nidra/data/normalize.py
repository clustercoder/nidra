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
        self.reference_std_: np.ndarray | None = None

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
        # Reference per-feature std of the TRAINING population, in the same
        # scaled+clipped units everything else is compared in. This is what
        # eval/metrics.state_nrmse should normalize by — a stable, millions-
        # of-rows statistic — rather than recomputing std on a small,
        # stratified eval batch, which collapses toward zero for the many
        # structurally near-constant features here (packet aggregates on a
        # flow-only day, rare-event ratios like urg_ratio) and blows up the
        # nRMSE ratio by orders of magnitude for reasons that have nothing to
        # do with forecast quality.
        Xs = self.transform(X_train)
        self.reference_std_ = Xs.reshape(-1, Xs.shape[-1]).std(axis=0)
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

    # Ceiling on the log1p-space value (post RobustScaler-inverse, pre-expm1)
    # for LOG1P_FEATURES before exponentiating back to raw units. expm1
    # amplifies error exponentially: a rollout prediction that is merely
    # "somewhat off" in scaled space (where state_nrmse is a sane ~5-7, see
    # eval/metrics.py) can invert to a raw-unit value in the 10^11-10^14
    # range with no ceiling — observed directly via reality_overlay.py
    # against the trained ensemble, not a hypothetical. 25 -> expm1(25) ~=
    # 7.2e10, generous for any of these five count features (bytes_total,
    # active_flow_count, out_degree, new_peer_count, retrans_count) even
    # under an extreme DDoS window, while keeping inverse_transform's output
    # numerically legible instead of a meaningless float.
    _LOG1P_INVERSE_CEIL = 25.0

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Undo `transform`: RobustScaler inverse, then expm1 on
        LOG1P_FEATURES. Used wherever a rolled-out or predicted state needs
        to be reported in the same raw units as the original input (e.g.
        `serve/predictor.py`'s `predicted_features`) — a forecast consumer
        should never have to know the model operates internally in scaled
        space. Lossy in two ways, both documented rather than hidden: (1)
        where `transform` already clipped to [clip_min, clip_max] — rare,
        RobustScaler already maps the typical range near [-3, 3]; (2) the
        pre-expm1 ceiling on LOG1P_FEATURES (see `_LOG1P_INVERSE_CEIL`) —
        without it, a rollout prediction that drifts by a modest amount in
        scaled space inverts to a physically nonsensical raw value (10^11+)
        because expm1 amplifies error exponentially. This bounds the raw
        OUTPUT to something legible; it does not change the scaled-space
        state the model actually operates on or is scored in."""
        if not self.fitted:
            raise RuntimeError("FeatureScaler.inverse_transform called before fit()/load()")
        if X_scaled.shape[-1] != len(FEATURE_ORDER):
            raise ValueError(f"expected {len(FEATURE_ORDER)} features, got {X_scaled.shape[-1]}")
        orig_shape = X_scaled.shape
        flat = np.asarray(X_scaled, dtype="float64").reshape(-1, orig_shape[-1])
        Xr = self.scaler.inverse_transform(flat).reshape(orig_shape)
        for name in LOG1P_FEATURES:
            idx = FEATURE_INDEX[name]
            clamped = np.clip(Xr[..., idx], a_min=None, a_max=self._LOG1P_INVERSE_CEIL)
            Xr[..., idx] = np.expm1(clamped)
        return Xr

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
            "reference_std": self.reference_std_.tolist(),
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
        # Older artifacts saved before reference_std_ existed: fall back to
        # None, which tells eval/metrics.state_nrmse to use the (less
        # robust) per-batch std instead of raising a hard error.
        ref = meta.get("reference_std")
        obj.reference_std_ = np.asarray(ref, dtype="float64") if ref is not None else None
        return obj
