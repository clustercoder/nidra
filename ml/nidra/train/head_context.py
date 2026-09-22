"""Rollout context for OBSERVED states: what a history-aware risk head is
trained on.

A `TrajectoryRiskHead` reads, at each rollout step k, the predicted state,
the encoder's hidden state after ingesting it, the transition's predicted
delta, and its predicted log-variance. Under the frozen-head discipline
(CLAUDE.md invariant 1) the head may only be TRAINED on observed states, so
each of those four has to be built from observations alone:

    state    the observed S_tau, scaled            (already in HeadArrays)
    hidden   encoder(S[tau-L+1 .. tau])            this module
    delta    S_tau - S_{tau-1}, in scaled space    this module
    logvar   transition(h_{tau-1}, S_{tau-1}, S_{tau-2}).logvar   this module

Every one of them is a function of windows at or before tau. Nothing here
reads row tau+1, and `test_head_context.py` pins that by tampering with the
tail of a table and asserting the head of the context is bit-identical.

Two implementation notes that are load-bearing rather than incidental.

The hidden state is computed over exactly L trailing windows, not over the
host's whole sequence. L is what the dynamics model was trained with and
what a rollout is given, so a hidden state accumulated over 4000 windows
would be a different quantity than the one the head meets at inference.
Sequences shorter than L are LEFT-padded with the scaler's silent state —
which is exactly what a host looks like before its first activity — rather
than with the first observation repeated.

Encoding N sequences of length L costs N*L recurrent steps and the [N, L, F]
tensor is never materialized whole: a real split is 2.3M rows, which at
L=30 would be 12 GB of float32. Rows are processed in chunks and only the
[N, H] / [N, F] outputs are kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER
from nidra.models.world_model import WorldModel

logger = logging.getLogger(__name__)

#: Rows encoded per pass. At L=30, F=45 one chunk of 8192 is a 44 MB float32
#: window tensor — small enough to be irrelevant, large enough that the GRU
#: call dominates the Python overhead.
DEFAULT_CHUNK_ROWS = 8192


@dataclass(frozen=True)
class HeadContext:
    """Per-row observed analogue of the rollout context, in HeadArrays order
    (sorted by host_id, window_ts)."""

    hidden: np.ndarray      # [N, H] encoder state after ingesting S_tau
    delta: np.ndarray       # [N, F] S_tau - S_{tau-1}, scaled space
    logvar: np.ndarray      # [N, F] frozen transition's log-variance for this step
    host_id: np.ndarray     # [N] str
    window_ts: np.ndarray   # [N] int64

    def __len__(self) -> int:
        return len(self.hidden)

    def as_components(self) -> dict[str, np.ndarray]:
        return {"hidden": self.hidden, "delta": self.delta, "logvar": self.logvar}


def _host_windows(scaled: np.ndarray, L: int, zero_state: np.ndarray) -> np.ndarray:
    """[n, L, F] trailing-L windows for one host's chronological sequence,
    left-padded with the silent state."""
    n, F = scaled.shape
    padded = np.vstack([np.tile(zero_state, (L - 1, 1)).astype(scaled.dtype), scaled])
    # stride_tricks over the padded array: no copy until the caller slices it
    windows = np.lib.stride_tricks.sliding_window_view(padded, (L, F))[:, 0]
    return windows[:n]


@torch.no_grad()
def build_head_context(table: pd.DataFrame, scaler: FeatureScaler, model: WorldModel, L: int,
                       chunk_rows: int = DEFAULT_CHUNK_ROWS, device: str = "cpu") -> HeadContext:
    """Observed rollout context for every row of a labelled state table.

    Row order matches `train.head_data.build_head_arrays` exactly (sorted by
    host_id then window_ts), because the two are consumed as one training
    sample per index and a mismatch would be silent.
    """
    H = model.encoder.hidden_size
    F = len(FEATURE_ORDER)
    if table.empty:
        e = np.zeros((0, F), dtype="float32")
        return HeadContext(np.zeros((0, H), dtype="float32"), e, e,
                           np.array([], dtype=object), np.zeros(0, dtype="int64"))

    df = table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
    scaled_all = scaler.transform(df[FEATURE_ORDER].to_numpy(dtype="float32")).astype("float32")
    zero_state = scaler.zero_state_scaled().astype("float32")

    hidden = np.zeros((len(df), H), dtype="float32")
    delta = np.zeros((len(df), F), dtype="float32")
    logvar = np.zeros((len(df), F), dtype="float32")

    model.eval()
    host_codes = df["host_id"].to_numpy()
    # Rows are contiguous per host after the sort, so boundaries are enough.
    boundaries = np.flatnonzero(np.r_[True, host_codes[1:] != host_codes[:-1]])
    bounds = np.r_[boundaries, len(df)]

    for lo, hi in zip(bounds[:-1], bounds[1:]):
        scaled = scaled_all[lo:hi]
        windows = _host_windows(scaled, L, zero_state)
        # S_{tau-1} and S_{tau-2}, read out of the padded window so the first
        # rows of a host need no special case.
        prev1 = windows[:, -2, :] if L >= 2 else np.tile(zero_state, (len(scaled), 1))
        prev2 = windows[:, -3, :] if L >= 3 else np.tile(zero_state, (len(scaled), 1))
        delta[lo:hi] = scaled - prev1

        for c0 in range(0, len(scaled), chunk_rows):
            c1 = min(c0 + chunk_rows, len(scaled))
            w = torch.from_numpy(np.ascontiguousarray(windows[c0:c1])).to(device)
            out, _ = model.encoder.gru(w)
            hidden[lo + c0: lo + c1] = out[:, -1, :].cpu().numpy()
            # The transition's uncertainty about THIS step, conditioned on
            # everything strictly before it: the encoder over the window's
            # first L-1 entries, plus the two states before tau. Reading it
            # out of the same pass is what keeps this identical to
            # WorldModel.observed_context, which has only the L-window to
            # work from at evaluation time — a head trained against one
            # definition and applied against the other would differ by a
            # window of history with nothing to show for it.
            h_prev = out[:, -2, :] if L >= 2 else torch.zeros_like(out[:, -1, :])
            _, lv = model.transition(
                h_prev,
                torch.from_numpy(np.ascontiguousarray(prev1[c0:c1])).to(device),
                torch.from_numpy(np.ascontiguousarray(prev2[c0:c1])).to(device))
            logvar[lo + c0: lo + c1] = lv.cpu().numpy()

    logger.info("build_head_context: %d rows over %d hosts at L=%d", len(df), len(bounds) - 1, L)
    return HeadContext(hidden=hidden, delta=delta, logvar=logvar,
                       host_id=host_codes.astype(object), window_ts=df["window_ts"].to_numpy(dtype="int64"))
