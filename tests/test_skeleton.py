"""P0 smoke tests: the package layout imports and the config loader reads defaults."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from nidra_common.config import DEV_SECRET_KEY, load_config

PACKAGES = [
    "api",
    "nidra",
    "nidra.data",
    "nidra.serve",
    "nidra_common",
    "services",
    "services.features",
    "services.inference",
    "services.ingest",
    "services.persister",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_config_defaults() -> None:
    cfg = load_config()
    assert cfg["window_delta"] == 30
    assert cfg["context_L"] == 30
    assert cfg["horizon_K"] == 6
    assert cfg["n_features"] == 45
    assert cfg["risk_threshold"] == 0.75
    assert cfg["lead_time_m"] == 2
    assert cfg["episode_close_after"] == 4
    assert cfg["predictor"]["impl"] == "stub"
    assert cfg["auth"]["algorithm"] == "HS256"
    assert set(cfg["streams"]) == {
        "raw_events",
        "state_vectors",
        "forecasts",
        "ingest_jobs",
        "maxlen",
    }
    assert cfg["ingest"]["allowed_ext"] == [".pcap", ".pcapng", ".csv"]


def test_env_overrides_urls_and_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIDRA_REDIS_URL", "redis://redis:6379/3")
    monkeypatch.setenv("NIDRA_POSTGRES_URL", "postgresql+asyncpg://u:p@db:5432/x")
    monkeypatch.setenv("NIDRA_SECRET_KEY", "not-the-dev-key")
    cfg = load_config()
    assert cfg["redis"]["url"] == "redis://redis:6379/3"
    assert cfg["postgres"]["url"] == "postgresql+asyncpg://u:p@db:5432/x"
    assert cfg["auth"]["secret_key"] == "not-the-dev-key"


def test_missing_secret_falls_back_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("NIDRA_SECRET_KEY", raising=False)
    with caplog.at_level("WARNING", logger="nidra_common.config"):
        cfg = load_config()
    assert cfg["auth"]["secret_key"] == DEV_SECRET_KEY
    assert any("SECRET_KEY" in record.message for record in caplog.records)


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
