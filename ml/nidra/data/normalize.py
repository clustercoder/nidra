"""Feature normalization: feature-specific transforms + statistics fit on the
TRAINING split only (and, within it, on ACTIVE windows only), serialized
alongside the model, never refit at serving or evaluation time.

Why this replaced a bare RobustScaler (reports/NIDRA_REEVALUATION_2026-09-19.md §8):
the training population is ~98% all-zero rows (empty windows), so the 25th
and 75th percentiles of every feature were both 0, sklearn substituted a
scale of 1, and the "scaler" that shipped was the identity — twelve heavy-
tailed features then saturated the ±10 clip on most active rows and were
effectively binary to the model. The fix has three parts, each explicit
per feature in `schema.FEATURE_TRANSFORMS`:

  1. a variance-stabilizing transform where the raw distribution is heavy-
     tailed: log1p for non-negative counts/durations/variances, asinh for
     signed deltas/slopes of those; nothing for bounded rates and entropies;
  2. z-scoring fit on the ACTIVE training rows, so the statistics describe
     traffic rather than silence. An all-zero row maps to a fixed point
     (-mean/std on the scaled features, 0 on the unit-range ones) — a
     distinct location for "silent", not the centre of the space;
  3. constant and duplicate features are detected on the training data
     and DROPPED (output forced to 0, excluded from the dynamics loss and
     the state-forecast metrics through `model_mask`). FEATURE_ORDER stays
     45 wide — the schema is a cross-service contract — the model simply
     never sees a column that carries no information.

The enforcement against leakage is structural, as before: `fit` is only ever
called from the training pipeline on train rows, and `transform` refuses to
run before `fit`/`load`. Everything the transform does is written to a plain
JSON file, so the serving side can reproduce it without pickled objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nidra.data.schema import FEATURE_INDEX, FEATURE_ORDER, FEATURE_TRANSFORMS, SCHEMA_VERSION

VALID_KINDS = ("log1p", "asinh", "zscore", "unit", "drop")

#: Ceiling on the log1p/asinh-space value before exponentiating back to raw
#: units in `inverse_transform`. expm1 amplifies error exponentially: a rollout
#: prediction that is merely "somewhat off" in scaled space can invert to a
#: raw-unit value in the 10^11+ range with no ceiling (observed with the Δ=30
#: ensemble via reality_overlay.py). 25 -> expm1(25) ~ 7.2e10 is generous for
#: any count/duration/variance feature here, even under a DDoS window.
_LOG_INVERSE_CEIL = 25.0

#: Floor on the per-feature std used for z-scoring. After the log/asinh
#: transforms every scaled feature has std O(1) on active rows; the floor only
#: guards a degenerate fit on a tiny synthetic training set.
_SCALE_FLOOR = 1e-3

#: Absolute correlation above which a feature is an exact duplicate of an
#: earlier one on the training data (post-transform, active rows).
_DUPLICATE_ABS_CORR = 0.999

#: Std below which a feature is constant on active training rows.
_CONSTANT_STD = 1e-8

#: The activity indicator is constant (1) on active rows by construction and
#: is the one feature exempt from the constant-feature drop.
ACTIVITY_FEATURE = "is_active"

#: Name of the serialized parameter file inside a scaler directory.
SCALER_PARAMS_FILENAME = "feature_scaler.json"
SCALER_METADATA_FILENAME = "scaler_metadata.json"


def apply_kind(x: np.ndarray, kind: str) -> np.ndarray:
    """The variance-stabilizing part of one feature's transform."""
    if kind == "log1p":
        return np.log1p(np.clip(x, a_min=0.0, a_max=None))
    if kind == "asinh":
        return np.arcsinh(x)
    if kind in ("zscore", "unit", "drop"):
        return x
    raise ValueError(f"unknown transform kind {kind!r}, expected one of {VALID_KINDS}")


def invert_kind(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log1p":
        return np.expm1(np.clip(z, a_min=None, a_max=_LOG_INVERSE_CEIL))
    if kind == "asinh":
        return np.sinh(np.clip(z, a_min=-_LOG_INVERSE_CEIL, a_max=_LOG_INVERSE_CEIL))
    return z


@dataclass
class FeatureScaler:
    clip_min: float = -10.0
    clip_max: float = 10.0
    kinds: list[str] = field(default_factory=lambda: [FEATURE_TRANSFORMS[f] for f in FEATURE_ORDER])
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    reference_std_: np.ndarray | None = None
    drop_reason_: dict[str, str] = field(default_factory=dict)
    n_fit_rows_: int = 0
    n_fit_active_rows_: int = 0
    fitted: bool = False

    # ------------------------------------------------------------------ fit
    def _check_width(self, X: np.ndarray) -> None:
        if X.shape[-1] != len(FEATURE_ORDER):
            raise ValueError(f"expected {len(FEATURE_ORDER)} features, got {X.shape[-1]}")

    def _stabilize(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype="float64").copy()
        for i, kind in enumerate(self.kinds):
            Xt[..., i] = apply_kind(Xt[..., i], kind)
        return Xt

    def fit(self, X_train: np.ndarray, active_mask: np.ndarray | None = None) -> "FeatureScaler":
        """Fit on TRAINING data only. `X_train` is [N, 45] raw states.
        Statistics are computed on the active rows (`active_mask`, or rows
        with any non-zero feature when not given); constant and duplicate
        features are detected on the same rows and dropped."""
        self._check_width(X_train)
        flat = np.asarray(X_train, dtype="float64").reshape(-1, X_train.shape[-1])
        if active_mask is None:
            active_mask = np.abs(flat).sum(axis=1) > 0
        active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
        if active_mask.sum() < 2:
            raise ValueError("FeatureScaler.fit needs at least two active training rows")
        act = self._stabilize(flat[active_mask])

        n = len(FEATURE_ORDER)
        center = np.zeros(n)
        scale = np.ones(n)
        kinds = list(self.kinds)
        reasons: dict[str, str] = {}
        mean = act.mean(axis=0)
        std = act.std(axis=0)

        for i, name in enumerate(FEATURE_ORDER):
            kind = kinds[i]
            if kind == "drop":
                reasons[name] = "declared"
                continue
            if std[i] < _CONSTANT_STD and name != ACTIVITY_FEATURE:
                # Constant on every active row: carries nothing beyond the
                # activity indicator itself (reciprocity ≡ 1 on active rows
                # in CIC-IDS2017 is the measured case).
                kinds[i] = "drop"
                reasons[name] = f"constant on active training rows (value {mean[i]:.4g}); redundant with {ACTIVITY_FEATURE}"
                continue
            if kind == "unit":
                continue  # already on a bounded scale: no centering, no scaling
            center[i] = mean[i]
            scale[i] = max(float(std[i]), _SCALE_FLOOR)

        # exact duplicates: compare every kept feature against the earlier kept
        # ones; the LATER feature in FEATURE_ORDER is the one dropped.
        kept = [i for i, k in enumerate(kinds) if k not in ("drop",)]
        if len(kept) > 1:
            z = (act[:, kept] - act[:, kept].mean(axis=0)) / np.clip(act[:, kept].std(axis=0), _SCALE_FLOOR, None)
            corr = (z.T @ z) / len(z)
            for a_pos in range(len(kept)):
                for b_pos in range(a_pos):
                    ia, ib = kept[a_pos], kept[b_pos]
                    if kinds[ia] == "drop" or kinds[ib] == "drop":
                        continue
                    if abs(corr[a_pos, b_pos]) > _DUPLICATE_ABS_CORR:
                        kinds[ia] = "drop"
                        reasons[FEATURE_ORDER[ia]] = f"duplicate of {FEATURE_ORDER[ib]} (|corr|={abs(corr[a_pos, b_pos]):.4f})"
                        center[ia], scale[ia] = 0.0, 1.0
                        break

        self.kinds = kinds
        self.center_ = center
        self.scale_ = scale
        self.drop_reason_ = reasons
        self.n_fit_rows_ = int(len(flat))
        self.n_fit_active_rows_ = int(active_mask.sum())
        self.fitted = True
        # Reference per-feature std of the whole TRAINING population in scaled
        # units — the stable normalizer eval/metrics.state_nrmse divides by.
        Xs = self.transform(flat)
        self.reference_std_ = Xs.std(axis=0)
        return self

    # ------------------------------------------------------------ transform
    @property
    def model_mask(self) -> np.ndarray:
        """bool [45]: True for features the model should read/predict/score;
        False for dropped ones (constant 0 in scaled space)."""
        return np.array([k != "drop" for k in self.kinds], dtype=bool)

    @property
    def dropped_features(self) -> list[str]:
        return [f for f, k in zip(FEATURE_ORDER, self.kinds) if k == "drop"]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureScaler.transform called before fit()/load() — never refit at serving time")
        self._check_width(X)
        Xt = self._stabilize(X)
        Xs = (Xt - self.center_) / self.scale_
        Xs = np.clip(Xs, self.clip_min, self.clip_max)
        Xs[..., ~self.model_mask] = 0.0
        return Xs

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Undo `transform` for reporting a predicted state in raw units.
        Lossy where `transform` clipped, and dropped features come back as
        their training-mean raw value is unknown, so as 0. Bounded by
        `_LOG_INVERSE_CEIL` before exponentiation (see its comment)."""
        if not self.fitted:
            raise RuntimeError("FeatureScaler.inverse_transform called before fit()/load()")
        self._check_width(X_scaled)
        Z = np.asarray(X_scaled, dtype="float64") * self.scale_ + self.center_
        out = Z.copy()
        for i, kind in enumerate(self.kinds):
            out[..., i] = invert_kind(Z[..., i], kind)
        out[..., ~self.model_mask] = 0.0
        return out

    def zero_state_scaled(self) -> np.ndarray:
        """Where an all-zero (silent) raw window lands in scaled space."""
        return self.transform(np.zeros((1, len(FEATURE_ORDER))))[0]

    # ----------------------------------------------------------- persistence
    @staticmethod
    def default_paths(scaler_dir: str | Path) -> tuple[Path, Path]:
        d = Path(scaler_dir)
        return d / SCALER_PARAMS_FILENAME, d / SCALER_METADATA_FILENAME

    def to_dict(self) -> dict:
        return {
            "feature_order": FEATURE_ORDER,
            "kinds": self.kinds,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "drop_reason": self.drop_reason_,
            "n_fit_rows": self.n_fit_rows_,
            "n_fit_active_rows": self.n_fit_active_rows_,
        }

    def save(self, scaler_path: str | Path, metadata_path: str | Path, extra_metadata: dict | None = None) -> None:
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted FeatureScaler")
        scaler_path, metadata_path = Path(scaler_path), Path(metadata_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        meta = {
            "feature_order": FEATURE_ORDER,
            "transforms": dict(zip(FEATURE_ORDER, self.kinds)),
            "dropped_features": self.dropped_features,
            "drop_reason": self.drop_reason_,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "schema_version": SCHEMA_VERSION,
            "method": "feature_transforms+zscore_on_active_rows",
            "fit_rows": self.n_fit_rows_,
            "fit_active_rows": self.n_fit_active_rows_,
            "reference_std": self.reference_std_.tolist(),
            "zero_state_scaled": self.zero_state_scaled().tolist(),
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
        with open(scaler_path) as f:
            params = json.load(f)
        if params.get("feature_order") != FEATURE_ORDER:
            raise ValueError("Serialized scaler parameters do not match FEATURE_ORDER — refusing to load.")
        obj = cls(clip_min=float(meta["clip_min"]), clip_max=float(meta["clip_max"]), kinds=list(params["kinds"]))
        obj.center_ = np.asarray(params["center"], dtype="float64")
        obj.scale_ = np.asarray(params["scale"], dtype="float64")
        obj.drop_reason_ = dict(params.get("drop_reason", {}))
        obj.n_fit_rows_ = int(params.get("n_fit_rows", 0))
        obj.n_fit_active_rows_ = int(params.get("n_fit_active_rows", 0))
        obj.fitted = True
        ref = meta.get("reference_std")
        obj.reference_std_ = np.asarray(ref, dtype="float64") if ref is not None else None
        return obj
