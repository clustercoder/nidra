"""P10: explanation, the model-internal what-if, the evaluation page, and model info.

Against the real Postgres, because the thing that could actually be wrong here is the
context rebuild: `GET /api/v1/explain/{host}/{ts}` has to hand the predictor the same
`[L, 45]` window the inference worker held at `ts`, read back out of the table the
features worker writes. A mocked session would assert that a query is issued, not that
the rows come back in the order the encoder reads them.

The rows are written through `StateVectorStore` — the writer the features worker uses —
so writer and reader are tested against each other rather than against a hand-built
fixture that could agree with neither.

Two assertions here are about claims rather than mechanics, and they are the ones that
matter on demo day: the counterfactual response carries the literal string
"model-internal what-if", and `/api/v1/benchmarks` says `pending` rather than inventing
numbers when the evaluation has not run.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from api import explain as explain_api
from api.main import create_app
from nidra.data.schema import FEATURE_ORDER
from nidra_common.bus import Bus, create_redis
from nidra_common.config import get_config
from nidra_common.db import dispose_engine, get_sessionmaker
from nidra_common.events import RawEvent
from nidra_common.schemas import SCHEMA_VERSION, StateVector
from services.features.store import StateVectorStore, load_context
from services.features.worker import FeaturesWorker
from services.inference.stub_predictor import COUNTERFACTUAL_LABEL, MODEL_VERSION

PASSWORD = "correct-horse-battery"

#: Aligned to an absolute multiple of the 30 s window, as the features service emits them.
BASE = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

HOST = "192.168.10.50"
UNKNOWN_HOST = "192.168.10.99"

#: One of the four drivers the stub reasons about, so clamping it visibly bends the curve.
DRIVER = "syn_ratio"


# ------------------------------------------------------------------------- fixtures


@pytest.fixture
def cfg() -> dict:
    return get_config()


@pytest.fixture
def context_l(cfg: dict) -> int:
    return int(cfg["context_L"])


@pytest.fixture
def window_delta(cfg: dict) -> int:
    return int(cfg["window_delta"])


@pytest.fixture
def horizon_k(cfg: dict) -> int:
    return int(cfg["horizon_K"])


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = create_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    factory = get_sessionmaker()
    try:
        yield factory
    finally:
        await dispose_engine()


async def _register(org: str) -> dict[str, str]:
    email = f"p10-{uuid.uuid4().hex[:12]}@nidra.test"
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        created = await client.post(
            "/api/v1/auth/register",
            json={"email": email, "password": PASSWORD, "org_name": org},
        )
        assert created.status_code == 201, created.text
        tokens = await client.post(
            "/api/v1/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert tokens.status_code == 200, tokens.text
    return {"tenant_id": created.json()["tenant_id"], "token": tokens.json()["access_token"]}


async def _purge(tenant_id: str) -> None:
    async with get_sessionmaker()() as session:
        for statement in (
            "DELETE FROM state_vectors WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM forecasts WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM users WHERE tenant_id = CAST(:tid AS uuid)",
            "DELETE FROM tenants WHERE id = CAST(:tid AS uuid)",
        ):
            await session.execute(text(statement), {"tid": tenant_id})
        await session.commit()
    await dispose_engine()


@pytest.fixture
async def account() -> AsyncIterator[dict[str, str]]:
    """A registered tenant with a live access token, purged afterwards."""
    created = await _register("P10 SOC")
    yield created
    await _purge(created["tenant_id"])


@pytest.fixture
async def other_account() -> AsyncIterator[dict[str, str]]:
    """A second tenant, so 'filtered by tenant' can be asserted rather than assumed."""
    created = await _register("P10 Other SOC")
    yield created
    await _purge(created["tenant_id"])


@pytest.fixture
async def api(account: dict[str, str]) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://api.test",
        headers={"Authorization": f"Bearer {account['token']}"},
    ) as client:
        yield client


# --------------------------------------------------------------- synthetic observations


def state_vector(tenant_id: str, *, window: int, window_delta: int) -> StateVector:
    """One window of a host fanning out — the shape the surrogate risk leans on."""
    features = dict.fromkeys(FEATURE_ORDER, 0.0)
    features["is_active"] = 1.0
    step = window + 1
    features[DRIVER] = min(0.95, 0.03 * step)
    features["dst_port_entropy"] = 0.15 * step
    features["new_peer_count"] = 2.0 * step
    features["out_degree"] = 2.0 * step
    features["d_out_degree"] = 2.0
    features["bytes_total"] = 1000.0 * step
    return StateVector(
        tenant_id=tenant_id,
        host_id=HOST,
        window_ts=BASE + timedelta(seconds=window * window_delta),
        features=features,
    )


@pytest.fixture
async def observed(
    account: dict[str, str],
    sessions: async_sessionmaker[AsyncSession],
    context_l: int,
    window_delta: int,
) -> list[StateVector]:
    """L windows of escalation, written through the store the features worker uses."""
    store = StateVectorStore(sessions)
    vectors = [
        state_vector(account["tenant_id"], window=window, window_delta=window_delta)
        for window in range(context_l)
    ]
    for vector in vectors:
        assert await store.save(vector) is True
    return vectors


def origin_of(observed: list[StateVector]) -> str:
    """The newest observed window — the `origin_ts` a forecast there would carry."""
    return observed[-1].window_ts.isoformat()


# ------------------------------------------------------------------------- the store


async def test_a_redelivered_state_vector_is_stored_once(
    observed: list[StateVector], sessions: async_sessionmaker[AsyncSession], context_l: int
) -> None:
    """At-least-once delivery writes the same window twice; the context must not grow."""
    store = StateVectorStore(sessions)
    assert await store.save(observed[-1]) is False

    async with sessions() as session:
        rows = await load_context(
            session,
            tenant_id=uuid.UUID(observed[-1].tenant_id),
            host_id=HOST,
            until=observed[-1].window_ts,
            limit=context_l * 2,
        )
    assert len(rows) == context_l
    assert [row.window_ts for row in rows] == [v.window_ts for v in observed], "not oldest-first"
    assert rows[-1].features == observed[-1].features


async def test_the_context_stops_at_the_requested_window(
    observed: list[StateVector], sessions: async_sessionmaker[AsyncSession], context_l: int
) -> None:
    """Nothing after `until` is read — the same causal boundary the forecast obeyed."""
    midpoint = observed[len(observed) // 2]
    async with sessions() as session:
        rows = await load_context(
            session,
            tenant_id=uuid.UUID(midpoint.tenant_id),
            host_id=HOST,
            until=midpoint.window_ts,
            limit=context_l,
        )
    assert rows[-1].window_ts == midpoint.window_ts
    assert all(row.window_ts <= midpoint.window_ts for row in rows)


async def test_the_features_worker_stores_every_vector_it_publishes(
    account: dict[str, str],
    sessions: async_sessionmaker[AsyncSession],
    redis_client: Redis,
    window_delta: int,
) -> None:
    """The wiring the explain endpoint rests on: published implies durable.

    A vector on `state_vectors` becomes a forecast, and a forecast the console can open
    but not explain is a hole in the product. Storing before publishing closes it, so
    this drives the real worker rather than the store alone.
    """
    tenant = account["tenant_id"]
    stream = f"test:state_vectors:{uuid.uuid4().hex[:12]}"
    bus = Bus(redis_client, stream, group="inference", consumer="test")
    worker = FeaturesWorker(redis_client, bus, cfg=get_config(), store=StateVectorStore(sessions))

    def event(ts: datetime) -> RawEvent:
        return RawEvent(
            tenant_id=tenant,
            job_id="p10",
            kind="flow",
            ts=ts,
            src_ip=HOST,
            dst_ip="10.0.0.7",
            dst_port=443,
            protocol=6,
            fields={"duration": 1.5, "bytes_fwd": 720.0, "pkts_fwd": 6.0, "syn_count": 4.0},
        )

    try:
        await worker.ingest_event(event(BASE))
        # An event in the next window closes the first one and publishes its vectors.
        emitted = await worker.ingest_event(event(BASE + timedelta(seconds=window_delta)))
        assert emitted, "the window did not close"

        async with sessions() as session:
            stored = await load_context(
                session,
                tenant_id=uuid.UUID(tenant),
                host_id=HOST,
                until=emitted[0].window_ts,
                limit=1,
            )
        assert [v.features for v in stored] == [emitted[0].features]
    finally:
        keys = [key async for key in redis_client.scan_iter(match=f"feat:*{tenant}*")]
        if keys:
            await redis_client.delete(*keys)
        await redis_client.delete(stream)


# --------------------------------------------------------------------------- explain


async def test_explain_returns_attributions_over_the_stored_context(
    api: httpx.AsyncClient, observed: list[StateVector], context_l: int, horizon_k: int
) -> None:
    response = await api.get(f"/api/v1/explain/{HOST}/{origin_of(observed)}")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["host_id"] == HOST
    assert body["horizon_k"] == horizon_k, "defaults to the end of the cone"
    assert body["model_version"] == MODEL_VERSION
    assert body["context_windows"] == context_l == body["context_l"]

    assert body["top_signals"], "an explanation with no attributed signals explains nothing"
    assert {s["name"] for s in body["top_signals"]} <= set(FEATURE_ORDER)
    ranked = [abs(s["shap_value"]) for s in body["top_signals"]]
    assert ranked == sorted(ranked, reverse=True), "signals are not strongest-first"

    assert len(body["window_importance"]) == context_l
    assert sum(body["window_importance"]) == pytest.approx(1.0, abs=1e-3)
    assert 0 <= body["driving_window"] < context_l


async def test_explain_flags_no_flows_until_the_flow_bridge_lands(
    api: httpx.AsyncClient, observed: list[StateVector]
) -> None:
    """The deliverable is stated as missing rather than quietly omitted or faked."""
    body = (await api.get(f"/api/v1/explain/{HOST}/{origin_of(observed)}")).json()
    assert body["flagged_flows"]["flows"] == []
    assert body["flagged_flows"]["available_after"] == "flow_bridge"
    assert body["flagged_flows"]["note"]


async def test_explain_accepts_an_earlier_horizon_step(
    api: httpx.AsyncClient, observed: list[StateVector]
) -> None:
    body = (await api.get(f"/api/v1/explain/{HOST}/{origin_of(observed)}", params={"k": 1})).json()
    assert body["horizon_k"] == 1


async def test_explain_refuses_a_step_past_the_horizon(
    api: httpx.AsyncClient, observed: list[StateVector], horizon_k: int
) -> None:
    response = await api.get(
        f"/api/v1/explain/{HOST}/{origin_of(observed)}", params={"k": horizon_k + 1}
    )
    assert response.status_code == 422, response.text


async def test_explain_404s_when_that_window_was_never_observed(
    api: httpx.AsyncClient, observed: list[StateVector], window_delta: int
) -> None:
    """Explaining a window with no stored state would explain a different forecast."""
    unobserved = (observed[-1].window_ts + timedelta(seconds=window_delta)).isoformat()
    assert (await api.get(f"/api/v1/explain/{HOST}/{unobserved}")).status_code == 404
    assert (
        await api.get(f"/api/v1/explain/{UNKNOWN_HOST}/{origin_of(observed)}")
    ).status_code == 404


async def test_explain_is_filtered_by_tenant(
    observed: list[StateVector], other_account: dict[str, str]
) -> None:
    """Tenant B cannot explain tenant A's host, by name or otherwise."""
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://api.test",
        headers={"Authorization": f"Bearer {other_account['token']}"},
    ) as client:
        response = await client.get(f"/api/v1/explain/{HOST}/{origin_of(observed)}")
        assert response.status_code == 404, "another tenant's state must not be readable"


# --------------------------------------------------------------------- counterfactual


async def test_the_counterfactual_is_labelled_model_internal_what_if(
    api: httpx.AsyncClient, observed: list[StateVector]
) -> None:
    """CLAUDE.md invariant. The literal string, in the response, every time."""
    response = await api.post(
        "/api/v1/counterfactual",
        json={
            "host": HOST,
            "ts": origin_of(observed),
            "feature": DRIVER,
            "clamp_value": 0.0,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["label"] == "model-internal what-if" == COUNTERFACTUAL_LABEL
    assert body["note"], "the response says what the label means, not just that it has one"


async def test_clamping_a_driver_flat_bends_the_curve_down(
    api: httpx.AsyncClient, observed: list[StateVector], horizon_k: int, window_delta: int
) -> None:
    body = (
        await api.post(
            "/api/v1/counterfactual",
            json={"host": HOST, "ts": origin_of(observed), "feature": DRIVER, "clamp_value": 0.0},
        )
    ).json()

    original = body["original"]
    modified = body["counterfactual"]
    assert [p["k"] for p in original] == list(range(1, horizon_k + 1))
    assert [p["k"] for p in modified] == list(range(1, horizon_k + 1))

    origin = observed[-1].window_ts
    assert modified[0]["ts"].startswith((origin + timedelta(seconds=window_delta)).isoformat()[:19])
    assert all(
        m["p_compromise"] < o["p_compromise"] for m, o in zip(modified, original, strict=True)
    ), "holding the SYN ratio at zero must lower the projected risk at every step"


async def test_an_unknown_feature_is_a_422(
    api: httpx.AsyncClient, observed: list[StateVector]
) -> None:
    """`FEATURE_ORDER` is the whole vocabulary; a name outside it never reaches the model."""
    response = await api.post(
        "/api/v1/counterfactual",
        json={
            "host": HOST,
            "ts": origin_of(observed),
            "feature": "not_a_feature",
            "clamp_value": 0.0,
        },
    )
    assert response.status_code == 422, response.text
    assert "FEATURE_ORDER" in response.text


async def test_the_counterfactual_404s_on_an_unobserved_window(
    api: httpx.AsyncClient, observed: list[StateVector], window_delta: int
) -> None:
    unobserved = (observed[-1].window_ts + timedelta(seconds=window_delta)).isoformat()
    response = await api.post(
        "/api/v1/counterfactual",
        json={"host": HOST, "ts": unobserved, "feature": DRIVER, "clamp_value": 0.0},
    )
    assert response.status_code == 404, response.text


# ------------------------------------------------------------------ benchmarks, model


async def test_benchmarks_is_pending_rather_than_zero_before_the_evaluation_runs(
    api: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No artifacts is a fact about the evaluation, not a set of numbers to invent."""
    monkeypatch.setattr(explain_api, "metrics_dir", lambda: tmp_path / "empty")
    body = (await api.get("/api/v1/benchmarks")).json()
    assert body["status"] == "pending"
    assert body["detail"] == "evaluation artifacts not yet produced"
    assert body["metrics"] == {}


async def test_benchmarks_serves_the_artifacts_verbatim(
    api: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact: dict[str, Any] = {
        "model": {"auc_k6": 0.71},
        "persistence": {"auc_k6": 0.52},
    }
    (tmp_path / "baselines.json").write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(explain_api, "metrics_dir", lambda: tmp_path)

    body = (await api.get("/api/v1/benchmarks")).json()
    assert body["status"] == "ok"
    assert body["metrics"] == {"baselines": artifact}, "artifacts are served as written"


async def test_an_unreadable_artifact_is_not_served_as_a_partial_result(
    api: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "good.json").write_text(json.dumps({"auc": 0.7}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(explain_api, "metrics_dir", lambda: tmp_path)

    response = await api.get("/api/v1/benchmarks")
    assert response.status_code == 500
    assert "broken.json" in response.text


async def test_the_configured_metrics_dir_is_the_one_served(cfg: dict) -> None:
    """Nothing hardcoded: the directory comes from `api.metrics_dir`."""
    assert explain_api.metrics_dir().is_absolute()
    assert str(explain_api.metrics_dir()).endswith(str(cfg["api"]["metrics_dir"]))


async def test_model_reports_the_predictor_and_the_geometry(
    api: httpx.AsyncClient, cfg: dict
) -> None:
    body = (await api.get("/api/v1/model")).json()
    assert body["impl"] == cfg["predictor"]["impl"]
    assert body["model_version"] == MODEL_VERSION
    assert body["schema_ver"] == SCHEMA_VERSION
    assert body["config"] == {
        "window_delta": cfg["window_delta"],
        "context_L": cfg["context_L"],
        "horizon_K": cfg["horizon_K"],
        "n_features": cfg["n_features"],
        "risk_threshold": cfg["risk_threshold"],
        "lead_time_m": cfg["lead_time_m"],
    }
    assert body["config"]["n_features"] == len(FEATURE_ORDER)


# ---------------------------------------------------------------------------- tenancy


async def test_every_endpoint_requires_a_token(observed: list[StateVector]) -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://api.test"
    ) as client:
        explained = await client.get(f"/api/v1/explain/{HOST}/{origin_of(observed)}")
        assert explained.status_code == 401
        assert (await client.get("/api/v1/benchmarks")).status_code == 401
        assert (await client.get("/api/v1/model")).status_code == 401
        posted = await client.post(
            "/api/v1/counterfactual",
            json={"host": HOST, "ts": origin_of(observed), "feature": DRIVER, "clamp_value": 0.0},
        )
        assert posted.status_code == 401
