"""Automated preprocessing audit, written next to every fitted scaler.

For each of the 45 features, on the ACTIVE training rows the scaler was fit
on: raw distribution summary (range, skew, zero fraction, distinct values),
the declared transform kind, the post-transform skew and clip-saturation
rate, the same two numbers under each alternative kind that is applicable
(`zscore` always; `log1p` if the feature is non-negative; `asinh` always),
the most correlated other feature, and the drop decision with its reason.

A feature is FLAGGED when its declared kind is not the best of the
applicable alternatives by |skew| (with a tolerance) or when it saturates
the clip on more than 1% of active rows — the audit does not change the
kind (that is a schema decision), it makes the disagreement visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nidra.data.normalize import FeatureScaler, apply_kind
from nidra.data.schema import FEATURE_ORDER

_SKEW_TOLERANCE = 0.5
_SATURATION_FLAG = 0.01


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    return float((((x - x.mean()) / sd) ** 3).mean())


def _scaled_stats(x_t: np.ndarray, clip: float) -> tuple[float, float]:
    """(skew, saturation) of a transformed feature after z-scoring."""
    sd = x_t.std()
    if sd < 1e-12:
        return 0.0, 0.0
    z = (x_t - x_t.mean()) / sd
    return _skew(z), float((np.abs(z) >= clip).mean())


def audit_features(X_train: np.ndarray, scaler: FeatureScaler, active_mask: np.ndarray | None = None) -> dict[str, Any]:
    flat = np.asarray(X_train, dtype="float64").reshape(-1, X_train.shape[-1])
    if active_mask is None:
        active_mask = np.abs(flat).sum(axis=1) > 0
    act = flat[np.asarray(active_mask, dtype=bool).reshape(-1)]
    clip = float(scaler.clip_max)
    Z_act = scaler.transform(act)

    # correlation partner search on the scaled, kept features
    kept = np.where(scaler.model_mask)[0]
    zk = Z_act[:, kept]
    zk = (zk - zk.mean(axis=0)) / np.clip(zk.std(axis=0), 1e-9, None)
    corr = (zk.T @ zk) / max(len(zk), 1)
    np.fill_diagonal(corr, 0.0)

    rows: list[dict[str, Any]] = []
    n_flagged = 0
    for i, name in enumerate(FEATURE_ORDER):
        x = act[:, i]
        kind = scaler.kinds[i]
        nonneg = bool((x >= 0).all())
        alternatives: dict[str, dict[str, float]] = {}
        for alt in ("zscore", "log1p", "asinh"):
            if alt == "log1p" and not nonneg:
                continue
            sk, sat = _scaled_stats(apply_kind(x, alt), clip)
            alternatives[alt] = {"abs_skew": abs(sk), "saturation": sat}
        chosen_eval = "zscore" if kind in ("unit", "drop") else kind
        chosen = alternatives.get(chosen_eval)
        best_alt = min(alternatives, key=lambda k: alternatives[k]["abs_skew"]) if alternatives else None
        flags: list[str] = []
        if kind not in ("drop", "unit") and chosen is not None and best_alt is not None:
            if alternatives[best_alt]["abs_skew"] + _SKEW_TOLERANCE < chosen["abs_skew"]:
                flags.append(f"{best_alt} would reduce |skew| {chosen['abs_skew']:.2f} -> {alternatives[best_alt]['abs_skew']:.2f}")
        sat_actual = float((np.abs(Z_act[:, i]) >= clip).mean()) if kind != "drop" else 0.0
        if sat_actual > _SATURATION_FLAG:
            flags.append(f"clip saturation {sat_actual:.3f} on active rows")
        partner = None
        if i in kept:
            pos = int(np.where(kept == i)[0][0])
            j = int(np.argmax(np.abs(corr[pos])))
            partner = {"feature": FEATURE_ORDER[int(kept[j])], "abs_corr": float(abs(corr[pos, j]))}
        rows.append({
            "feature": name,
            "kind": kind,
            "dropped": kind == "drop",
            "drop_reason": scaler.drop_reason_.get(name),
            "raw": {
                "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()),
                "abs_skew": abs(_skew(x)), "zero_fraction": float((x == 0).mean()),
                "n_unique": int(len(np.unique(x))), "non_negative": nonneg,
            },
            "scaled": {"center": float(scaler.center_[i]), "scale": float(scaler.scale_[i]),
                       "abs_skew": abs(_skew(Z_act[:, i])) if kind != "drop" else None,
                       "saturation": sat_actual,
                       "silent_row_value": float(scaler.zero_state_scaled()[i])},
            "alternatives": alternatives,
            "most_correlated": partner,
            "flags": flags,
        })
        n_flagged += bool(flags)
    return {
        "n_active_rows": int(len(act)),
        "n_rows": int(len(flat)),
        "clip": clip,
        "dropped_features": scaler.dropped_features,
        "drop_reason": scaler.drop_reason_,
        "n_flagged": n_flagged,
        "features": rows,
    }


def audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Preprocessing audit",
        "",
        f"Fit rows: {report['n_rows']:,} (active: {report['n_active_rows']:,}). Clip: ±{report['clip']:g}. "
        f"Dropped: {', '.join(report['dropped_features']) or 'none'}. Flagged: {report['n_flagged']}.",
        "",
        "| feature | kind | raw range | raw \\|skew\\| | zero frac | scaled \\|skew\\| | saturation | silent-row value | most correlated | flags |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in report["features"]:
        raw, sc = r["raw"], r["scaled"]
        partner = f"{r['most_correlated']['feature']} ({r['most_correlated']['abs_corr']:.2f})" if r["most_correlated"] else "—"
        skew_s = f"{sc['abs_skew']:.2f}" if sc["abs_skew"] is not None else "—"
        kind = r["kind"] + (f" ({r['drop_reason']})" if r["dropped"] else "")
        lines.append(
            f"| {r['feature']} | {kind} | [{raw['min']:.3g}, {raw['max']:.3g}] | {raw['abs_skew']:.2f} | {raw['zero_fraction']:.2f} "
            f"| {skew_s} | {sc['saturation']:.3f} | {sc['silent_row_value']:.2f} | {partner} | {'; '.join(r['flags']) or ''} |"
        )
    return "\n".join(lines) + "\n"


def write_audit(X_train: np.ndarray, scaler: FeatureScaler, out_dir: str | Path,
                active_mask: np.ndarray | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = audit_features(X_train, scaler, active_mask)
    with open(out_dir / "preprocessing_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    (out_dir / "preprocessing_audit.md").write_text(audit_markdown(report))
    return report
