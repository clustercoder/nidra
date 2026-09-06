"""What travels on `ingest_jobs` and `raw_events`.

Two files enter the system — a CICFlowMeter CSV and a PCAP — and they describe the same
network from opposite ends: a flow is an aggregate over a conversation, a packet is one
observation of it. Rather than teach the features service both dialects, ingest
normalises each into a :class:`RawEvent`: the same five-tuple envelope, a `kind`
discriminator, and a `fields` payload holding only the measurements that kind carries.

`fields` is `dict[str, float]` and is validated against the vocabulary for its kind, so a
typo'd key fails here — at the producer — rather than becoming a silently-zero feature
three services downstream. It is a *subset* check, not equality: a UDP packet has no
`tcp_len`, and inventing one would be worse than its absence.

The CIC `Label` column is deliberately **not** in `fields`. It is a string, it is
supervision rather than measurement, and it is carried on its own optional attribute so
that nothing in the serving path can mistake it for an input.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, model_validator

SCHEMA_VERSION = "1.0"

#: Flow-side measurements, from the CICFlowMeter columns. Byte and packet counts are
#: directional (`fwd` = away from the source host); durations and inter-arrival times are
#: **seconds**, converted from the CSV's microseconds at parse time.
FLOW_FIELDS: frozenset[str] = frozenset(
    {
        "duration",
        "bytes_fwd",
        "bytes_bwd",
        "pkts_fwd",
        "pkts_bwd",
        "syn_count",
        "ack_count",
        "rst_count",
        "fin_count",
        "psh_count",
        "urg_count",
        "iat_mean",
        "iat_std",
        "iat_max",
    }
)

#: Packet-side measurements, one per `tshark -T fields` row (IMPLEMENTATION-ML.md §2.2).
#: `frag` combines the more-fragments bit with a non-zero fragment offset; `retransmission`
#: is 1.0 only when tshark set `tcp.analysis.retransmission`, which is empty for normal
#: packets. TCP-only fields are absent on UDP rows rather than zero.
PACKET_FIELDS: frozenset[str] = frozenset(
    {
        "ttl",
        "tcp_window",
        "frag",
        "frame_len",
        "tcp_len",
        "tcp_flags",
        "retransmission",
    }
)

FIELD_VOCABULARY: dict[str, frozenset[str]] = {
    "flow": FLOW_FIELDS,
    "packet": PACKET_FIELDS,
}

#: File kinds the ingest worker knows how to parse. `.pcapng` uploads are `pcap` jobs —
#: tshark reads both and the parse path is identical.
JobKind = Literal["csv", "pcap"]

#: Job lifecycle, as reported by `ingest_jobs.status` and the `job:{job_id}` Redis hash.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_COMPLETE = "complete"
JOB_ERROR = "error"


class RawEvent(BaseModel):
    """One normalised record on `raw_events`: a flow summary or a single packet."""

    tenant_id: str
    job_id: str
    kind: Literal["flow", "packet"]
    ts: datetime  # capture time, UTC; a naive value is read as UTC downstream
    src_ip: str
    dst_ip: str
    src_port: int | None = None
    dst_port: int | None = None
    protocol: int | None = None  # IP protocol number: 6 TCP, 17 UDP
    fields: dict[str, float]
    label: str | None = None  # CIC ground truth, supervision only — never an input
    schema_ver: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _fields_are_known_and_finite(self) -> RawEvent:
        vocabulary = FIELD_VOCABULARY[self.kind]
        unknown = sorted(set(self.fields) - vocabulary)
        if unknown:
            raise ValueError(f"{self.kind} event has unknown fields: {unknown}")
        if not self.fields:
            raise ValueError(f"{self.kind} event carries no fields")
        non_finite = sorted(k for k, v in self.fields.items() if not math.isfinite(v))
        if non_finite:
            # A NaN here becomes a NaN feature, which becomes a NaN forecast.
            raise ValueError(f"{self.kind} event has non-finite fields: {non_finite}")
        return self


class IngestJobMessage(BaseModel):
    """The `ingest_jobs` entry: the API wrote a validated file, a worker should replay it.

    `path` is server-side and was constructed from the job id — never from the uploaded
    filename, which is retained only for display.
    """

    job_id: str
    tenant_id: str
    kind: JobKind
    path: str
    filename: str
    speed: float  # replay multiplier; 0 publishes as fast as the bus accepts
    schema_ver: str = SCHEMA_VERSION
