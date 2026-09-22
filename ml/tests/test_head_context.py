"""Rollout context for OBSERVED states — the training side of a
history-aware head.

At inference the head reads (predicted state, encoder hidden state,
predicted delta, predicted log-variance) at each rollout step. Under the
frozen-head discipline it must be TRAINED on the observed analogue of each
of those, and the analogue has to be built without reading anything after
the row it describes. These tests pin both halves: that each quantity is the
right one, and that it is causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from nidra.data.normalize import FeatureScaler
from nidra.data.schema import FEATURE_ORDER
from nidra.models.world_model import WorldModel
from nidra.train.head_context import build_head_context


@pytest.fixture
def model() -> WorldModel:
    torch.manual_seed(0)
    m = WorldModel(n_features=45, hidden_size=8, encoder_layers=1, transition_mlp_hidden=16)
    m.eval()
    return m


@pytest.fixture
def scaler() -> FeatureScaler:
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(300, 45))) + 0.2
    return FeatureScaler().fit(X, active_mask=np.ones(300, dtype=bool))


def _table(n_hosts: int = 2, n_windows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for h in range(n_hosts):
        for i in range(n_windows):
            row = {name: float(abs(rng.normal())) for name in FEATURE_ORDER}
            row.update(host_id=f"h{h}", window_ts=60 * i, stage_label="benign", risk_label=0)
            rows.append(row)
    return pd.DataFrame(rows)


class TestShapesAndAlignment:
    def test_one_row_of_context_per_row_of_the_table(self, model, scaler):
        table = _table()
        ctx = build_head_context(table, scaler, model, L=5)
        assert ctx.hidden.shape == (len(table), model.encoder.hidden_size)
        assert ctx.delta.shape == (len(table), 45)
        assert ctx.logvar.shape == (len(table), 45)

    def test_rows_follow_the_head_array_order(self, model, scaler):
        # build_head_arrays sorts by (host_id, window_ts); the context must
        # use the same order or every training pair is mismatched.
        table = _table().sample(frac=1.0, random_state=1).reset_index(drop=True)
        ctx = build_head_context(table, scaler, model, L=5)
        expected = table.sort_values(["host_id", "window_ts"]).reset_index(drop=True)
        assert list(ctx.host_id) == list(expected["host_id"])
        assert list(ctx.window_ts) == list(expected["window_ts"])

    def test_chunking_changes_nothing(self, model, scaler):
        table = _table(n_hosts=3, n_windows=20)
        a = build_head_context(table, scaler, model, L=6, chunk_rows=4)
        b = build_head_context(table, scaler, model, L=6, chunk_rows=10_000)
        assert np.allclose(a.hidden, b.hidden, atol=1e-5)
        assert np.allclose(a.logvar, b.logvar, atol=1e-5)
        assert np.allclose(a.delta, b.delta, atol=1e-5)


class TestCausality:
    def test_the_hidden_state_is_the_encoder_over_the_trailing_L_windows(self, model, scaler):
        table = _table(n_hosts=1, n_windows=12)
        L = 5
        ctx = build_head_context(table, scaler, model, L=L)
        scaled = scaler.transform(table[FEATURE_ORDER].to_numpy(dtype="float32"))
        for tau in (L - 1, 7, 11):
            window = torch.from_numpy(scaled[tau - L + 1: tau + 1]).float().unsqueeze(0)
            with torch.no_grad():
                h_t, _ = model.encoder(window)
            assert np.allclose(ctx.hidden[tau], h_t.numpy()[0], atol=1e-5), tau

    def test_early_rows_are_left_padded_with_the_silent_state(self, model, scaler):
        table = _table(n_hosts=1, n_windows=12)
        L = 5
        ctx = build_head_context(table, scaler, model, L=L)
        scaled = scaler.transform(table[FEATURE_ORDER].to_numpy(dtype="float32"))
        zero = scaler.zero_state_scaled()
        window = np.vstack([np.tile(zero, (L - 1, 1)), scaled[0:1]]).astype("float32")
        with torch.no_grad():
            h_t, _ = model.encoder(torch.from_numpy(window).unsqueeze(0))
        assert np.allclose(ctx.hidden[0], h_t.numpy()[0], atol=1e-5)

    def test_changing_a_later_row_cannot_change_an_earlier_context(self, model, scaler):
        table = _table(n_hosts=1, n_windows=12)
        before = build_head_context(table, scaler, model, L=5)
        tampered = table.copy()
        tampered.loc[9:, FEATURE_ORDER] = 7.0
        after = build_head_context(tampered, scaler, model, L=5)
        assert np.allclose(before.hidden[:9], after.hidden[:9], atol=1e-6)
        assert np.allclose(before.logvar[:9], after.logvar[:9], atol=1e-6)
        assert np.allclose(before.delta[:9], after.delta[:9], atol=1e-6)

    def test_a_host_never_sees_another_hosts_history(self, model, scaler):
        two = _table(n_hosts=2, n_windows=12)
        one = two[two["host_id"] == "h1"].reset_index(drop=True)
        ctx_two = build_head_context(two, scaler, model, L=5)
        ctx_one = build_head_context(one, scaler, model, L=5)
        h1 = ctx_two.host_id == "h1"
        assert np.allclose(ctx_two.hidden[h1], ctx_one.hidden, atol=1e-6)


class TestQuantities:
    def test_delta_is_the_backward_difference_in_scaled_space(self, model, scaler):
        table = _table(n_hosts=1, n_windows=8)
        ctx = build_head_context(table, scaler, model, L=4)
        scaled = scaler.transform(table[FEATURE_ORDER].to_numpy(dtype="float32"))
        assert np.allclose(ctx.delta[3], scaled[3] - scaled[2], atol=1e-5)
        assert np.allclose(ctx.delta[0], scaled[0] - scaler.zero_state_scaled(), atol=1e-5)

    def test_logvar_is_the_frozen_transitions_prediction_for_this_step(self, model, scaler):
        table = _table(n_hosts=1, n_windows=8)
        L = 4
        ctx = build_head_context(table, scaler, model, L=L)
        scaled = scaler.transform(table[FEATURE_ORDER].to_numpy(dtype="float32"))
        tau = 6
        # h_prev is the encoder over the window's first L-1 entries — the
        # same quantity WorldModel.observed_context can build from an
        # [L, F] history alone. See head_context.py for why it is that and
        # not a full L-window ending at tau-1.
        window = torch.from_numpy(scaled[tau - L + 1: tau]).float().unsqueeze(0)
        with torch.no_grad():
            h_prev, _ = model.encoder(window)
            _, logvar = model.transition(h_prev,
                                          torch.from_numpy(scaled[tau - 1]).float().unsqueeze(0),
                                          torch.from_numpy(scaled[tau - 2]).float().unsqueeze(0))
        assert np.allclose(ctx.logvar[tau], logvar.numpy()[0], atol=1e-5)

    def test_context_is_finite_even_for_a_single_window_host(self, model, scaler):
        table = _table(n_hosts=1, n_windows=1)
        ctx = build_head_context(table, scaler, model, L=5)
        assert np.isfinite(ctx.hidden).all() and np.isfinite(ctx.logvar).all() and np.isfinite(ctx.delta).all()

    def test_empty_table_returns_empty_context(self, model, scaler):
        ctx = build_head_context(_table().iloc[0:0], scaler, model, L=5)
        assert len(ctx.hidden) == 0 and len(ctx.delta) == 0
