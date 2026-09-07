"""Generator for `tests/fixtures/replay.csv` — the capture `make demo` replays.

Three hosts over `context_L + horizon_K` windows and change, in the published
CICFlowMeter layout (leading-space column names and day-first timestamps included), so
the demo exercises exactly the parser a real Wednesday slice would.

* one **escalating** host: quiet for the first stretch, then a horizontal scan whose
  fan-out, new peers and destination-port spread all climb window over window while its
  SYN ratio sits at 0.5 — the four drivers the predictor reads.
* two **benign** hosts: the same two connections to the same server every window.

The point of the shape is the comparison. The escalating host's projected risk has to
separate from the benign hosts', and it has to do so from features that move, not from a
label — the `Label` column is written for a human reading the file and is never an input
(DECISIONS.md D24).

The window count is read from `config/default.yaml`: a context of L windows produces no
forecast at all until the L-th, so a fixture shorter than L + K would replay in full and
forecast nothing. Regenerate after changing `context_L`, `horizon_K` or `window_delta`:

    make fixtures        # or: python -m tests.fixtures.replay_csv
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from nidra_common.config import REPO_ROOT, get_config

#: Where the generated file lives, and what `demo.fixture` in the config points at.
DEFAULT_PATH = REPO_ROOT / "tests" / "fixtures" / "replay.csv"

#: Wednesday morning, to match the capture the project's splits describe. Arbitrary but
#: fixed: the replay clock paces on the *gaps* between timestamps, not their absolute value.
CAPTURE_START = datetime(2017, 7, 5, 9, 0, 0, tzinfo=UTC)

ESCALATING_HOST = "192.168.10.50"
BENIGN_HOSTS: tuple[str, ...] = ("192.168.10.60", "192.168.10.61")
SERVER = "192.168.10.3"

#: Windows beyond L + K. The first forecast is only possible once L windows have been
#: observed, so this is the room the escalation has to develop *in front of an audience*
#: rather than inside the context that produced the first cone.
EXTRA_WINDOWS = 16

#: Where the scan starts, as a fraction of the capture. Chosen so it begins *after* the
#: L-th window: a scan that starts earlier is already in the context of the first
#: forecast, which then opens above the threshold and there is no crossing to watch.
RAMP_START_FRACTION = 0.66

#: Scan fan-out: this many new hosts on the first window of the ramp, and this many more
#: on each window after it. It never plateaus — a fan-out that stops growing flattens
#: every slope the risk drifts on, and the point of the fixture is a host still climbing
#: when the last forecast is made.
FANOUT_BASE = 2
FANOUT_GROWTH = 6

#: Leading spaces on every column but the first: the published CIC layout, verbatim.
CSV_HEADER = (
    "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol,"
    " Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,"
    " Total Length of Fwd Packets, Total Length of Bwd Packets, Flow IAT Mean,"
    " Flow IAT Std, Flow IAT Max, FIN Flag Count, SYN Flag Count, RST Flag Count,"
    " PSH Flag Count, ACK Flag Count, URG Flag Count, Label"
)

TCP = 6


def window_count(cfg: dict | None = None) -> int:
    """`context_L + horizon_K + EXTRA_WINDOWS`, from config rather than a literal."""
    config = cfg if cfg is not None else get_config()
    return int(config["context_L"]) + int(config["horizon_K"]) + EXTRA_WINDOWS


def _row(
    *,
    flow_id: str,
    src: str,
    src_port: int,
    dst: str,
    dst_port: int,
    ts: datetime,
    pkts_fwd: int,
    pkts_bwd: int,
    bytes_fwd: int,
    bytes_bwd: int,
    syn: int,
    ack: int,
    label: str,
) -> tuple[datetime, str]:
    """One CSV line, paired with its timestamp so the file can be written in order."""
    fields = [
        flow_id,
        src,
        str(src_port),
        dst,
        str(dst_port),
        str(TCP),
        # CIC writes day-first local time with no timezone.
        ts.strftime("%-d/%-m/%Y %H:%M:%S"),
        "1500000" if pkts_bwd else "40000",  # flow duration, microseconds
        str(pkts_fwd),
        str(pkts_bwd),
        str(bytes_fwd),
        str(bytes_bwd),
        "250000" if pkts_bwd else "0",  # flow IAT mean
        "1000" if pkts_bwd else "0",  # flow IAT std
        "400000" if pkts_bwd else "0",  # flow IAT max
        "1" if pkts_bwd else "0",  # FIN
        str(syn),
        "0",  # RST
        "1" if pkts_bwd else "0",  # PSH
        str(ack),
        "0",  # URG
        label,
    ]
    return ts, ",".join(fields)


def _benign_rows(host: str, window: int, start: datetime) -> list[tuple[datetime, str]]:
    """Two established connections to the same server: high ACK, one SYN, low SYN ratio."""
    return [
        _row(
            flow_id=f"benign-{host}-{window}-{index}",
            src=host,
            src_port=40000 + window * 2 + index,
            dst=SERVER,
            dst_port=port,
            ts=start + timedelta(seconds=2 + index * 5),
            pkts_fwd=10,
            pkts_bwd=12,
            bytes_fwd=1200,
            bytes_bwd=9800,
            syn=1,
            ack=9,
            label="BENIGN",
        )
        for index, port in enumerate((443, 80))
    ]


def _target(index: int) -> str:
    """A scan target, spread over 10.10.x.y so a long sweep never repeats an address."""
    return f"10.10.{1 + index // 254}.{1 + index % 254}"


def _scan_rows(window: int, start: datetime, fanout: int, first_target: int) -> list:
    """`fanout` SYN probes, each to a host and port neither seen before.

    One packet each and no reply, so `syn_ratio` is 1.0 — the shape of a half-open scan.
    Every peer is new, so `new_peer_count` and `out_degree` track the fan-out and
    `dst_port_entropy` rises as ln(fanout): a sweep, widening.
    """
    rows = []
    for index in range(fanout):
        target = first_target + index
        rows.append(
            _row(
                flow_id=f"scan-{window}-{index}",
                src=ESCALATING_HOST,
                src_port=50000 + index,
                dst=_target(target),
                dst_port=1024 + (target * 7) % 40000,
                ts=start + timedelta(seconds=index % 29),
                pkts_fwd=1,
                pkts_bwd=0,
                bytes_fwd=60,
                bytes_bwd=0,
                syn=1,
                ack=0,
                label="PortScan",
            )
        )
    return rows


def replay_rows(cfg: dict | None = None) -> list[str]:
    """Every CSV line of the fixture, in timestamp order."""
    config = cfg if cfg is not None else get_config()
    delta = int(config["window_delta"])
    windows = window_count(config)
    ramp_start = int(windows * RAMP_START_FRACTION)

    rows: list[tuple[datetime, str]] = []
    first_target = 0
    for window in range(windows):
        start = CAPTURE_START + timedelta(seconds=window * delta)
        for host in BENIGN_HOSTS:
            rows.extend(_benign_rows(host, window, start))
        # The escalating host keeps its two ordinary connections throughout: the scan is
        # added to normal traffic, not swapped for it, so `syn_ratio` climbs as the probes
        # come to outnumber the packets of a working host instead of stepping to 1.0 in
        # one window. A step is easy to forecast and nothing like a real escalation.
        rows.extend(_benign_rows(ESCALATING_HOST, window, start))
        if window < ramp_start:
            continue
        fanout = FANOUT_BASE + FANOUT_GROWTH * (window - ramp_start)
        rows.extend(_scan_rows(window, start, fanout, first_target))
        first_target += fanout

    rows.sort(key=lambda item: item[0])
    return [line for _, line in rows]


def write_replay_csv(path: Path | str = DEFAULT_PATH, cfg: dict | None = None) -> Path:
    """Write the fixture and return where it landed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join([CSV_HEADER, *replay_rows(cfg)]) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    config = get_config()
    path = write_replay_csv(cfg=config)
    print(
        f"wrote {path} — {window_count(config)} windows of "
        f"{int(config['window_delta'])}s, {len(replay_rows(config))} flows"
    )


if __name__ == "__main__":
    main()
