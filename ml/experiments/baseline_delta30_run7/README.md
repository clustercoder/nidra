# Δ=30 run-7 baseline artifacts (preserved copy)

Copied from `ml/artifacts/{weights,scaler}` on 2026-09-20 before the Δ=60 production
retrain overwrote them. Each `model_seed_*.pt` matches the sha256 recorded in
`../BASELINE_MANIFEST_delta30_run7.json` (verified at copy time). The scaler here is the
identity RobustScaler these weights were trained with (`robust_scaler.joblib`); it is not
loadable by the current `FeatureScaler` — evaluate these weights from the
`baseline-delta30-run7` git tag. Weights are gitignored; this directory is a disk copy.
