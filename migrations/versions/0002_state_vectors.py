"""Observed state vectors, kept so a forecast can be explained after the fact.

`GET /api/v1/explain/{host}/{ts}` has to hand the predictor the same `[L, 45]` context
the inference worker held at `ts`. That context lives in Redis while the replay runs
(`seq:{tenant}:{host}`), but it is a rolling window under a TTL, on a Redis started with
`--appendonly no --save ""` — it answers "explain the newest forecast" and nothing else.
So the features worker writes every vector it publishes here first, and the explain
endpoint reads the L rows ending at `ts` back out.

`UNIQUE (tenant_id, host_id, window_ts)` is the same idempotency device the forecasts
table uses: at-least-once redelivery means the same window will be written twice, and
`ON CONFLICT DO NOTHING` needs the constraint to exist. Its index is also the read path
— `(tenant_id, host_id, window_ts DESC) LIMIT L` walks it directly — so there is no
second index to keep.

Revision ID: 0002_state_vectors
Revises: 0001_initial_schema
Create Date: 2026-09-07
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_state_vectors"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "state_vectors",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("host_id", sa.Text(), nullable=False),
        sa.Column("window_ts", sa.TIMESTAMP(timezone=True), nullable=False),
        # The 45 features as the schema names them, not 45 columns: the feature set is
        # `FEATURE_ORDER`'s to change, and a migration per feature would make it the
        # database's. Reads project it back through `FEATURE_ORDER`, never dict order.
        sa.Column("features", postgresql.JSONB(), nullable=False),
        sa.Column("schema_ver", sa.Text()),
        sa.UniqueConstraint(
            "tenant_id",
            "host_id",
            "window_ts",
            name="uq_state_vectors_tenant_host_window_ts",
        ),
    )


def downgrade() -> None:
    op.drop_table("state_vectors")
