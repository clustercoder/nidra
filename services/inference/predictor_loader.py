"""Chooses the predictor the inference worker runs, and loads it exactly once.

Two implementations sit behind one switch in `config/default.yaml`:

```yaml
predictor:
  impl: stub    # stub | nidra
```

`stub` is the placeholder built for P7; `nidra` is the trained world model. The real one
is imported **lazily, inside the branch**, for two reasons. It drags in torch, and a
backend running the stub should not pay a multi-second import for a module it will not
call. More importantly, the import is `nidra.serve.predictor` and nothing else: the whole
ML surface the serving plane is allowed to touch is one class with three methods. The day
a service imports `nidra.models`, the boundary that keeps the model swappable is gone.

`torch.set_num_threads(2)` is applied here rather than in the worker because this is the
one place torch can be present. PyTorch defaults to every core, and four inference
replicas on one machine each grabbing all of them spend their time in the scheduler.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from nidra_common.config import REPO_ROOT, get_config
from services.inference.stub_predictor import StubPredictor

logger = logging.getLogger(__name__)

STUB_IMPL = "stub"
NIDRA_IMPL = "nidra"

#: Two threads per worker: enough for the ensemble to overlap, few enough that replicas
#: on one host do not thrash. IMPLEMENTATION-Backend.md §7.
TORCH_THREADS = 2


@runtime_checkable
class Predictor(Protocol):
    """The entire ML surface the serving plane depends on (IMPLEMENTATION-ML.md §7)."""

    def forecast(self, states: np.ndarray, host_id: str, origin_ts: datetime) -> dict[str, Any]:
        """`[L, F]` raw observed windows, oldest first → a `Forecast`-shaped dict."""

    def counterfactual(
        self, states: np.ndarray, feature_name: str, clamp_value: float
    ) -> dict[str, Any]:
        """Re-simulate with one feature clamped. Model-internal what-if, never causal."""

    def explain(self, states: np.ndarray, horizon_k: int) -> dict[str, Any]:
        """Attribution for the step-`horizon_k` projection."""


def configure_torch_threads(threads: int = TORCH_THREADS) -> bool:
    """Cap torch's thread pool if torch is installed. Returns whether it was applied.

    Guarded by import availability: the stub path has no torch dependency and the
    backend must stay installable without it.
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch is not installed; thread cap not applied")
        return False
    torch.set_num_threads(threads)
    logger.info("torch.set_num_threads(%d)", threads)
    return True


def _predictor_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("predictor", {}))


def _resolve(path: str) -> Path:
    """Config paths are repo-relative unless absolute, as everywhere else in the config."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (REPO_ROOT / candidate)


def load_predictor(cfg: dict[str, Any] | None = None) -> Predictor:
    """Build the configured predictor. Called once, at process start, never per message."""
    config = cfg if cfg is not None else get_config()
    predictor_cfg = _predictor_cfg(config)
    impl = str(predictor_cfg.get("impl", STUB_IMPL))

    configure_torch_threads()

    if impl == STUB_IMPL:
        logger.info("loading StubPredictor — placeholder until nidra.serve.predictor lands")
        return StubPredictor(cfg=config)

    if impl == NIDRA_IMPL:
        # Lazy and narrow: `nidra.serve.predictor` is the only ML module a service imports.
        from nidra.serve.predictor import NidraPredictor

        weights_dir = _resolve(str(predictor_cfg["weights_dir"]))
        scaler_path = _resolve(str(predictor_cfg["scaler_path"]))
        config_path = _resolve(str(predictor_cfg["config_path"]))
        logger.info("loading NidraPredictor from %s", weights_dir)
        return NidraPredictor(
            weights_dir=weights_dir,
            scaler_path=scaler_path,
            config_path=config_path,
        )

    raise ValueError(f"predictor.impl must be {STUB_IMPL!r} or {NIDRA_IMPL!r}, got {impl!r}")
