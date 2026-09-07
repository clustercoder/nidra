"""Initial serving-plane schema (IMPLEMENTATION-Backend.md §9).

Six tables: tenants, users, ingest_jobs, forecasts, episodes, benchmark_runs.

Two things here are load-bearing rather than incidental:

* ``forecasts`` stores the full :class:`Forecast` as JSONB with the hot fields promoted
  to columns, so schema evolution costs nothing while the UI's queries stay indexed.
* ``UNIQUE (tenant_id, host_id, origin_ts)`` is what makes the persister idempotent
  under at-least-once redelivery — ``ON CONFLICT DO NOTHING`` needs it to exist.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
        ),
        sa.Column("email", postgresql.CITEXT(), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), server_default=sa.text("'analyst'")),
    )

    op.create_table(
        "ingest_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id"),
        ),
        sa.Column("filename", sa.Text()),
        sa.Column("kind", sa.Text()),
        sa.Column("status", sa.Text()),
        sa.Column("progress", sa.REAL(), server_default=sa.text("0")),
        sa.Column("error", sa.Text()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "forecasts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("origin_ts", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("observed_risk", sa.REAL()),
        sa.Column("observed_stage", sa.Text()),
        sa.Column("max_p_compromise", sa.REAL()),
        sa.Column("lead_time_s", sa.REAL()),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("model_version", sa.Text()),
        sa.UniqueConstraint(
            "tenant_id",
            "host_id",
            "origin_ts",
            name="uq_forecasts_tenant_host_origin_ts",
        ),
    )
    op.create_index(
        "ix_forecasts_tenant_origin_ts",
        "forecasts",
        ["tenant_id", sa.text("origin_ts DESC")],
    )
    op.create_index(
        "ix_forecasts_tenant_host_origin_ts",
        "forecasts",
        ["tenant_id", "host_id", sa.text("origin_ts DESC")],
    )
    # Partial index: the alert query. Most rows are quiet hosts and never touch it.
    op.create_index(
        "ix_forecasts_tenant_max_p_compromise",
        "forecasts",
        ["tenant_id", sa.text("max_p_compromise DESC")],
        postgresql_where=sa.text("max_p_compromise > 0.5"),
    )

    op.create_table(
        "episodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True)),
        sa.Column("host_id", sa.Text()),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True)),
        sa.Column("peak_risk", sa.REAL()),
        sa.Column("stages", postgresql.ARRAY(sa.Text())),
        sa.Column("first_alert_at", sa.TIMESTAMP(timezone=True)),
        # Populated only when replaying a labelled dataset; this is what lets the UI
        # state "warned 90 s before onset" as a fact rather than an average.
        sa.Column("ground_truth_onset", sa.TIMESTAMP(timezone=True)),
    )

    op.create_table(
        "benchmark_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.Text()),
        sa.Column("dataset", sa.Text()),
        sa.Column("split", sa.Text()),
        sa.Column("metrics", postgresql.JSONB()),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("benchmark_runs")
    op.drop_table("episodes")
    op.drop_index("ix_forecasts_tenant_max_p_compromise", table_name="forecasts")
    op.drop_index("ix_forecasts_tenant_host_origin_ts", table_name="forecasts")
    op.drop_index("ix_forecasts_tenant_origin_ts", table_name="forecasts")
    op.drop_table("forecasts")
    op.drop_table("ingest_jobs")
    op.drop_table("users")
    op.drop_table("tenants")
