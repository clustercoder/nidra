"""NidraPredictor: the ONLY surface the backend imports.

The backend must never import nidra.models.*, nidra.data.*, or
nidra.explain.* directly — everything it needs is exposed through this
one class. Loads the ensemble weights and scaler ONCE at construction,
never per call. Stateless with respect to per-host sequence history:
`forecast()` takes the caller's own [L, F] window buffer as an argument
rather than holding one internally — sequence management belongs to the
backend/Redis layer (IMPLEMENTATION-Backend.md §7), not here, so any
inference worker can serve any host without coordination.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import (
    CONTEXT_LENGTH,
    FEATURE_ORDER,
    HORIZON_LENGTH,
    SCHEMA_VERSION,
    STAGE_LABELS,
    WINDOW_SECONDS,
    validate_state_array_width,
)
from nidra.explain.counterfactual import COUNTERFACTUAL_LABEL, counterfactual_rollout
from nidra.explain.saliency import temporal_saliency
from nidra.explain.shap_runner import explain_current_risk, explain_predicted_stage, load_background, top_signals
from nidra.models.world_model import WorldModel
from nidra.utils.config import load_config

logger = logging.getLogger(__name__)

MODEL_VERSION = "nidra-v0.1.0"


class NidraPredictor:
    """One class, loaded once, three methods: forecast, counterfactual,
    explain. See IMPLEMENTATION-ML.md §7 and IMPLEMENTATION-Backend.md §7
    for the exact contract the serving plane depends on."""

    def __init__(
        self,
        weights_dir: str | Path,
        scaler_path: str | Path,
        config_path: str | Path | None = None,
        seeds: list[int] | None = None,
        device: str = "cpu",
    ):
        self.cfg = load_config(config_path)
        self.device = device
        torch.set_num_threads(self.cfg.get("serving", {}).get("torch_num_threads", 2))

        weights_dir = Path(weights_dir)
        scaler_path = Path(scaler_path)
        metadata_path = scaler_path.parent / "scaler_metadata.json"
        self.scaler = FeatureScaler.load(scaler_path, metadata_path)

        background_path = scaler_path.parent / "shap_background.npy"
        self._background = load_background(background_path)
        if self._background is None:
            logger.warning(
                "NidraPredictor: no serialized SHAP background at %s — falling back to the "
                "per-call observed history as background (degraded: not the trained k-means "
                "benign centroids). Run train_dynamics to generate one.", background_path,
            )

        seeds = seeds if seeds is not None else self.cfg["ensemble"]["seeds"]
        self.models: list[WorldModel] = []
        for seed in seeds:
            model_path = weights_dir / f"model_seed_{seed}.pt"
            if not model_path.exists():
                logger.warning("NidraPredictor: missing weights for seed %d at %s, skipping", seed, model_path)
                continue
            model = self._build_model()
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.freeze_all()
            self.models.append(model)

        if not self.models:
            raise RuntimeError(f"NidraPredictor: no ensemble weights found in {weights_dir}")

        self.n_samples_per_member = self.cfg["rollout"]["n_samples_per_member"]
        self.K = self.cfg["windowing"]["horizon_length"]
        self.L = self.cfg["windowing"]["context_length"]
        self.risk_threshold = self.cfg["eval"]["risk_threshold"]
        logger.info("NidraPredictor: loaded %d ensemble member(s) from %s", len(self.models), weights_dir)

    def _build_model(self) -> WorldModel:
        mcfg = self.cfg["model"]
        return WorldModel(
            n_features=mcfg["n_features"],
            hidden_size=mcfg["encoder"]["hidden_size"],
            encoder_layers=mcfg["encoder"]["num_layers"],
            encoder_dropout=mcfg["encoder"]["dropout"],
            transition_mlp_hidden=mcfg["transition"]["mlp_hidden"],
            logvar_min=mcfg["transition"]["logvar_min"],
            logvar_max=mcfg["transition"]["logvar_max"],
            risk_hidden=mcfg["risk_head"]["hidden"],
            stage_hidden=mcfg["stage_head"]["hidden"],
            n_stages=mcfg["stage_head"]["n_stages"],
            state_clamp=mcfg["transition"]["state_clamp"],
        ).to(self.device)

    def _validate_and_scale(self, states: np.ndarray) -> np.ndarray:
        """states: [L, F] raw, unscaled, oldest-first. Fails loudly on any
        schema mismatch rather than producing a silently wrong forecast."""
        states = np.asarray(states, dtype="float32")
        if states.ndim != 2:
            raise ValueError(f"expected states of shape [L, F], got shape {states.shape}")
        validate_state_array_width(states.shape[1])
        if states.shape[0] != self.L:
            raise ValueError(f"expected {self.L} windows of history (L), got {states.shape[0]}")
        if not np.isfinite(states).all():
            raise ValueError("states contains NaN or Inf — refusing to forecast on invalid input")
        return self.scaler.transform(states)

    @torch.no_grad()
    def _ensemble_rollout(self, scaled_states: np.ndarray) -> dict:
        """Runs every ensemble member's rollout and pools trajectories
        before scoring — ~n_samples_per_member * len(models) trajectories
        total, matching the ~1000-trajectory target for a 5-seed ensemble
        at 200 samples/member."""
        x = torch.from_numpy(scaled_states).float().unsqueeze(0)  # [1, L, F]
        all_states = []
        for model in self.models:
            out = model.rollout(x, K=self.K, n_samples=self.n_samples_per_member, stochastic=True)
            all_states.append(out.states)  # [1, S, K, F]
        pooled_states = torch.cat(all_states, dim=1)  # [1, S_total, K, F]

        risk_list, stage_list = [], []
        for model in self.models:
            r, s = model.score_states(pooled_states.reshape(-1, self.K, pooled_states.shape[-1]))
            risk_list.append(r.reshape(1, pooled_states.shape[1], self.K))
            stage_list.append(s.reshape(1, pooled_states.shape[1], self.K, -1))
        # average head outputs across ensemble members too, not just samples
        risk = torch.stack(risk_list, dim=0).mean(dim=0)     # [1, S_total, K]
        stage = torch.stack(stage_list, dim=0).mean(dim=0)   # [1, S_total, K, n_stages]

        q_low = self.cfg["rollout"]["ci_low_quantile"]
        q_high = self.cfg["rollout"]["ci_high_quantile"]
        return {
            "predicted_states_mean": pooled_states.mean(dim=1)[0].numpy(),   # [K, F]
            "risk_mean_k": risk.mean(dim=1)[0].numpy(),                       # [K]
            "risk_ci_low_k": risk.quantile(q_low, dim=1)[0].numpy(),
            "risk_ci_high_k": risk.quantile(q_high, dim=1)[0].numpy(),
            "stage_mean_k": stage.mean(dim=1)[0].numpy(),                     # [K, n_stages]
            "n_trajectories": pooled_states.shape[1],
        }

    def forecast(self, states: np.ndarray, host_id: str, origin_ts: datetime) -> dict:
        """states: [L, F] raw, unscaled, oldest-first. Returns a dict
        matching the backend's shared `Forecast` schema (see
        IMPLEMENTATION-Backend.md §3) — `tenant_id` is NOT included here;
        the backend attaches it, since the ML layer has no notion of
        tenancy."""
        scaled = self._validate_and_scale(states)
        rollout = self._ensemble_rollout(scaled)

        observed_state = states[-1]
        with torch.no_grad():
            observed_risk_t, observed_stage_probs_t = self.models[0].score_states(
                torch.from_numpy(scaled[-1]).float().unsqueeze(0)
            )
        observed_risk = float(observed_risk_t.item())
        observed_stage = STAGE_LABELS[int(observed_stage_probs_t.argmax(dim=-1).item())]

        horizons = []
        window_seconds = self.cfg["windowing"]["window_seconds"]
        for k in range(self.K):
            ts = origin_ts + timedelta(seconds=window_seconds * (k + 1))
            stage_dist = {name: float(p) for name, p in zip(STAGE_LABELS, rollout["stage_mean_k"][k])}
            predicted_features = {name: float(v) for name, v in zip(FEATURE_ORDER, rollout["predicted_states_mean"][k])}
            horizons.append({
                "k": k + 1,
                "ts": ts.isoformat(),
                "p_compromise": float(rollout["risk_mean_k"][k]),
                "ci_low": float(rollout["risk_ci_low_k"][k]),
                "ci_high": float(rollout["risk_ci_high_k"][k]),
                "stage_dist": stage_dist,
                "predicted_features": predicted_features,
            })

        lead_time_s = self._lead_time_from_curve(horizons, window_seconds)

        background = self._shap_background(scaled)
        attributions = explain_current_risk(scaled[-1], background, self.models[0], nsamples=100)
        signals = [
            {"name": a["feature"], "shap_value": a["shap_value"], "direction": a["direction"],
             "display": self._human_readable_signal(a)}
            for a in top_signals(attributions, n=5)
        ]

        saliency = temporal_saliency(self.models[0], scaled, target_feature=FEATURE_ORDER[0], horizon_k=0, K=self.K)

        return {
            "host_id": host_id,
            "origin_ts": origin_ts.isoformat(),
            "horizons": horizons,
            "lead_time_s": lead_time_s,
            "observed_stage": observed_stage,
            "observed_risk": observed_risk,
            "top_signals": signals,
            "driving_window": saliency["driving_window"],
            "model_version": MODEL_VERSION,
            "schema_ver": SCHEMA_VERSION,
            "n_trajectories": rollout["n_trajectories"],
        }

    def counterfactual(self, states: np.ndarray, feature_name: str, clamp_value: float) -> dict:
        """Clamp one named feature to `clamp_value` for the entire rollout
        and re-simulate. Explicitly labelled 'model-internal what-if' — see
        explain/counterfactual.py. Never call this an intervention."""
        scaled = self._validate_and_scale(states)
        if feature_name not in FEATURE_ORDER:
            raise ValueError(f"unknown feature {feature_name!r}")
        clamped_scaled_value = self._scale_single_feature_value(feature_name, clamp_value)

        result = counterfactual_rollout(
            scaled, self.models[0], feature_name, clamped_scaled_value,
            K=self.K, n_samples=self.n_samples_per_member,
        )
        return {
            "label": COUNTERFACTUAL_LABEL,
            "clamped_feature": feature_name,
            "clamp_value": clamp_value,
            "risk_mean_k": result["risk_mean_k"][0].tolist(),
            "risk_ci_low_k": result["risk_ci_low_k"][0].tolist(),
            "risk_ci_high_k": result["risk_ci_high_k"][0].tolist(),
        }

    def explain(self, states: np.ndarray, horizon_k: int) -> dict:
        """Full attribution bundle for one host/window: SHAP on observed
        risk, temporal saliency, and SHAP on the predicted stage at
        `horizon_k`. Three distinct mechanisms, kept distinct in the
        response — see IMPLEMENTATION-ML.md §6."""
        scaled = self._validate_and_scale(states)
        background = self._shap_background(scaled)

        current_risk_attributions = explain_current_risk(scaled[-1], background, self.models[0], nsamples=100)

        with torch.no_grad():
            out = self.models[0].rollout(
                torch.from_numpy(scaled).float().unsqueeze(0), K=self.K, n_samples=1, stochastic=False,
            )
        predicted_state = out.states[0, 0, horizon_k, :].numpy()
        with torch.no_grad():
            _, stage_probs = self.models[0].score_states(torch.from_numpy(predicted_state).float().unsqueeze(0))
        predicted_stage_idx = int(stage_probs.argmax(dim=-1).item())
        stage_attributions = explain_predicted_stage(predicted_state, background, self.models[0], predicted_stage_idx, nsamples=100)

        saliency_results = {
            feat: temporal_saliency(self.models[0], scaled, target_feature=feat, horizon_k=horizon_k, K=self.K)
            for feat in [a["feature"] for a in current_risk_attributions[:3]]
        }

        return {
            "current_risk_attributions": top_signals(current_risk_attributions, n=10),
            "predicted_stage": STAGE_LABELS[predicted_stage_idx],
            "predicted_stage_attributions": top_signals(stage_attributions, n=10),
            "temporal_saliency": saliency_results,
            "horizon_k": horizon_k,
        }

    def _shap_background(self, scaled_history: np.ndarray) -> np.ndarray:
        """Uses the serialized k-means benign-centroid background computed
        at training time (see train/train_dynamics.py). Falls back to the
        observed history itself only when no background was serialized —
        a documented degradation, not the random background the spec
        explicitly warns is slow and noisy."""
        if self._background is not None:
            return self._background
        return scaled_history

    def _scale_single_feature_value(self, feature_name: str, raw_value: float) -> float:
        idx = FEATURE_ORDER.index(feature_name)
        dummy = np.zeros((1, len(FEATURE_ORDER)), dtype="float32")
        dummy[0, idx] = raw_value
        scaled = self.scaler.transform(dummy)
        return float(scaled[0, idx])

    @staticmethod
    def _human_readable_signal(attribution: dict) -> str:
        direction_word = "rising" if attribution["direction"] == "up" else "falling"
        return f"{attribution['feature']} {direction_word}"

    @staticmethod
    def _lead_time_from_curve(horizons: list[dict], window_seconds: int, threshold: float = 0.75, m: int = 2) -> float | None:
        run = 0
        for i, h in enumerate(horizons):
            run = run + 1 if h["p_compromise"] >= threshold else 0
            if run >= m:
                return float(window_seconds * (i - m + 2))
        return None
