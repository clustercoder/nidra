"""Redis Streams wrapper — the one bus abstraction every service uses.

Written once, imported everywhere, because the delivery semantics matter more than the
convenience: a message is acknowledged **only after** the handler that processed it
returned. On exception it stays in the consumer group's pending list, where
:meth:`Bus.reclaim` (``XAUTOCLAIM``) picks it up once it has been idle long enough. That
is a genuine at-least-once guarantee rather than a fire-and-forget queue.

Two properties this module exists to preserve:

* **Idempotent group creation** — ``XGROUP CREATE ... id="0" MKSTREAM``, tolerating
  ``BUSYGROUP``. A group created after the first messages were published starts at the
  tail and the worker sits idle next to a full stream.
* **Independent fan-out** — two *different* groups on one stream each receive every
  message, while consumers *within* one group split them. The WebSocket design in P9
  depends on the first half of that sentence and is broken by the second.

Stream names and ``maxlen`` come from ``config/default.yaml``; nothing here is hardcoded.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import ResponseError

from nidra_common.config import get_config

logger = logging.getLogger(__name__)

#: The single field every stream entry carries: the model's JSON, as produced by Pydantic.
PAYLOAD_FIELD = "payload"

#: Start of stream. Both the ``XGROUP CREATE`` id and the ``XAUTOCLAIM`` cursor sentinel.
NEW_GROUP_ID = "0"
SCAN_START = "0-0"

DEFAULT_COUNT = 32
DEFAULT_BLOCK_MS = 2000
DEFAULT_MIN_IDLE_MS = 60_000


def redis_url(cfg: dict[str, Any] | None = None) -> str:
    """Redis URL for this process (``NIDRA_REDIS_URL`` overrides the YAML)."""
    config = cfg if cfg is not None else get_config()
    url = config.get("redis", {}).get("url")
    if not url:
        raise ValueError("redis.url is missing from the NIDRA config")
    return str(url)


def stream_maxlen(cfg: dict[str, Any] | None = None) -> int:
    """Approximate cap on stream length, from config. Caps memory, never correctness."""
    config = cfg if cfg is not None else get_config()
    maxlen = config.get("streams", {}).get("maxlen")
    if not maxlen:
        raise ValueError("streams.maxlen is missing from the NIDRA config")
    return int(maxlen)


def stream_name(key: str, cfg: dict[str, Any] | None = None) -> str:
    """Resolve a logical stream key (``raw_events``, ``forecasts``, ...) to its name."""
    config = cfg if cfg is not None else get_config()
    name = config.get("streams", {}).get(key)
    if not name:
        raise ValueError(f"streams.{key} is missing from the NIDRA config")
    return str(name)


def create_redis(cfg: dict[str, Any] | None = None, **kwargs: Any) -> Redis:
    """Async Redis client on the configured URL.

    Responses stay as bytes: payloads are JSON that gets handed straight to
    ``model_validate_json``, and decoding them to `str` only to re-encode is waste.
    """
    return Redis.from_url(redis_url(cfg), decode_responses=False, **kwargs)


def _as_str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _payload(fields: dict[Any, Any]) -> bytes:
    """Extract the payload field, tolerating a client configured to decode responses."""
    for key in (PAYLOAD_FIELD.encode(), PAYLOAD_FIELD):
        if key in fields:
            value = fields[key]
            return value.encode() if isinstance(value, str) else bytes(value)
    raise KeyError(f"stream entry has no {PAYLOAD_FIELD!r} field: keys={list(fields)}")


class Bus:
    """One consumer's view of one stream: publish, read, ack, reclaim.

    `consumer` names *this process* within `group`. Two processes sharing a group split
    the stream between them (the inference scale-out unit); two processes on different
    groups each see all of it (`api` and `persister` on `forecasts`).
    """

    def __init__(
        self,
        redis: Redis,
        stream: str,
        group: str,
        consumer: str,
        maxlen: int | None = None,
    ) -> None:
        self.r = redis
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.maxlen = stream_maxlen() if maxlen is None else maxlen

    def __repr__(self) -> str:
        return f"Bus(stream={self.stream!r}, group={self.group!r}, consumer={self.consumer!r})"

    async def ensure_group(self) -> None:
        """Create the consumer group at the head of the stream. Safe to call repeatedly.

        ``id="0"`` so a worker started after the producer still sees earlier messages;
        ``mkstream=True`` so startup order between producer and consumer does not matter.
        """
        try:
            await self.r.xgroup_create(self.stream, self.group, id=NEW_GROUP_ID, mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
            logger.debug("consumer group %s already exists on %s", self.group, self.stream)

    async def publish(self, model: BaseModel) -> str:
        """Append one Pydantic model to the stream; returns the assigned message id."""
        msg_id = await self.r.xadd(
            self.stream,
            {PAYLOAD_FIELD: model.model_dump_json()},
            maxlen=self.maxlen,
            approximate=True,
        )
        return _as_str(msg_id)

    async def read(
        self, count: int = DEFAULT_COUNT, block_ms: int = DEFAULT_BLOCK_MS
    ) -> list[tuple[str, bytes]]:
        """One ``XREADGROUP`` poll. Returns `[]` when the block elapses with no messages.

        Delivered messages are now pending for this consumer until :meth:`ack`.
        """
        resp = await self.r.xreadgroup(
            self.group,
            self.consumer,
            {self.stream: ">"},
            count=count,
            block=block_ms,
        )
        return [
            (_as_str(msg_id), _payload(fields))
            for _, messages in resp or []
            for msg_id, fields in messages
        ]

    async def consume(
        self, count: int = DEFAULT_COUNT, block_ms: int = DEFAULT_BLOCK_MS
    ) -> AsyncIterator[tuple[str, bytes]]:
        """Yield ``(msg_id, payload_bytes)`` forever. The caller acks after processing."""
        while True:
            for item in await self.read(count=count, block_ms=block_ms):
                yield item

    async def ack(self, msg_id: str | bytes) -> int:
        """Acknowledge one message. Call this **only** after processing succeeded."""
        return int(await self.r.xack(self.stream, self.group, msg_id))

    async def reclaim(
        self, min_idle_ms: int = DEFAULT_MIN_IDLE_MS, count: int = DEFAULT_COUNT
    ) -> AsyncIterator[tuple[str, bytes]]:
        """Take over messages another consumer left pending for longer than `min_idle_ms`.

        This is the recovery half of at-least-once: a worker that crashed mid-handler
        never acked, so its messages are still in the group's pending list and someone
        has to claim them. Yields them exactly as :meth:`consume` does — still pending,
        now owned by this consumer, to be acked once processed.
        """
        cursor = SCAN_START
        while True:
            resp = await self.r.xautoclaim(
                self.stream,
                self.group,
                self.consumer,
                min_idle_time=min_idle_ms,
                start_id=cursor,
                count=count,
            )
            cursor, messages = _as_str(resp[0]), resp[1]
            for msg_id, fields in messages:
                if not fields:
                    # Trimmed out from under the pending entry; nothing to process.
                    logger.warning("dropping reclaimed empty entry %s on %s", msg_id, self.stream)
                    continue
                yield _as_str(msg_id), _payload(fields)
            if cursor == SCAN_START:
                return

    async def pending_count(self) -> int:
        """Unacknowledged messages across the whole group — the lag number `/health` shows."""
        summary = await self.r.xpending(self.stream, self.group)
        return int(summary["pending"]) if summary else 0
