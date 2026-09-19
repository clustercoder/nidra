"""MITRE ATT&CK mapping for NIDRA's stage taxonomy and for the CIC-IDS2017
attack labels the model is trained and evaluated on.

Two tables, one purpose: when the world model projects a host into a stage
bucket at horizon k, the console has to say which ATT&CK tactic (and, where
the dataset supports it, which technique) that bucket corresponds to — and
say how much of that is actually resolvable from network telemetry.

Both tables are CURATED, documented mappings, not learned ones, and they
are deliberately narrow:

  * The six-bucket stage taxonomy (`schema.STAGE_LABELS`) is what the
    frozen stage head emits. Each bucket maps to one or two tactics.
  * CIC-IDS2017 gives a per-flow attack *label*, not a technique. The
    technique column below is the closest ATT&CK technique for the tool the
    dataset authors ran (Patator, Hulk, slowloris, Ares, Nmap, ...), taken
    from the dataset paper's description of each attack — it is what an
    analyst would file the alert under, not something the model infers.
  * What the model can and cannot resolve is stated per bucket in
    `RESOLUTION_NOTES`. In particular, the `exfil` bucket in CIC-IDS2017 is
    occupied entirely by DoS/DDoS episodes (there is no exfiltration in the
    dataset), so a predicted `exfil` stage means "Impact-like terminal
    activity" on this training data, and the mapping says so.

ATT&CK identifiers are Enterprise ATT&CK v14 (October 2023).

`<repo>/docs/ATTACK_MAPPING.md` is generated from these tables by
`python -m nidra.data.attack_mapping --write`; a test checks the two agree.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from nidra.data.labels import map_label_to_stage
from nidra.data.schema import STAGE_LABELS

ATTACK_VERSION = "Enterprise ATT&CK v14"


@dataclass(frozen=True)
class Tactic:
    id: str
    name: str


@dataclass(frozen=True)
class Technique:
    id: str
    name: str
    tactic: Tactic


@dataclass(frozen=True)
class StageMapping:
    stage: str
    tactics: tuple[Tactic, ...]
    resolution: str  # what network telemetry can and cannot tell about this bucket


@dataclass(frozen=True)
class LabelMapping:
    label: str          # raw CIC-IDS2017 `Label` value
    day: str            # capture day it occurs on
    stage: str          # NIDRA bucket, via labels.map_label_to_stage
    techniques: tuple[Technique, ...]
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


# --- tactics ----------------------------------------------------------------
TA_RECON = Tactic("TA0043", "Reconnaissance")
TA_DISCOVERY = Tactic("TA0007", "Discovery")
TA_INITIAL_ACCESS = Tactic("TA0001", "Initial Access")
TA_CREDENTIAL_ACCESS = Tactic("TA0006", "Credential Access")
TA_LATERAL = Tactic("TA0008", "Lateral Movement")
TA_C2 = Tactic("TA0011", "Command and Control")
TA_EXFIL = Tactic("TA0010", "Exfiltration")
TA_IMPACT = Tactic("TA0040", "Impact")

# --- techniques (only those a CIC-IDS2017 label actually corresponds to) ---
T_SCAN_IP = Technique("T1595.001", "Active Scanning: Scanning IP Blocks", TA_RECON)
T_NET_SVC_DISCOVERY = Technique("T1046", "Network Service Discovery", TA_DISCOVERY)
T_PASSWORD_GUESS = Technique("T1110.001", "Brute Force: Password Guessing", TA_CREDENTIAL_ACCESS)
T_EXTERNAL_REMOTE = Technique("T1133", "External Remote Services", TA_INITIAL_ACCESS)
T_EXPLOIT_PUBLIC = Technique("T1190", "Exploit Public-Facing Application", TA_INITIAL_ACCESS)
T_REMOTE_SERVICES = Technique("T1021", "Remote Services", TA_LATERAL)
T_WEB_C2 = Technique("T1071.001", "Application Layer Protocol: Web Protocols", TA_C2)
T_NET_DOS_FLOOD = Technique("T1498.001", "Network Denial of Service: Direct Network Flood", TA_IMPACT)
T_SVC_EXHAUST = Technique("T1499.002", "Endpoint Denial of Service: Service Exhaustion Flood", TA_IMPACT)
T_APP_EXHAUST = Technique("T1499.003", "Endpoint Denial of Service: Application Exhaustion Flood", TA_IMPACT)
T_EXFIL_C2 = Technique("T1041", "Exfiltration Over C2 Channel", TA_EXFIL)


STAGE_MAPPINGS: dict[str, StageMapping] = {
    "benign": StageMapping("benign", (), "No tactic. The bucket every host sits in almost all of the time."),
    "recon": StageMapping(
        "recon", (TA_RECON, TA_DISCOVERY),
        "Resolvable from flow telemetry: fan-out to many destination ports/hosts with tiny payloads "
        "(dst_port_entropy, out_degree, new_peer_count, syn_ratio). External scanning is Reconnaissance; "
        "the same behaviour from an internal host is Discovery — the mapping cannot tell them apart "
        "without knowing which side of the perimeter the host is on."),
    "initial_access": StageMapping(
        "initial_access", (TA_INITIAL_ACCESS, TA_CREDENTIAL_ACCESS),
        "Brute-force logins (many short sessions to one service port) and web exploitation are visible in "
        "flow shape; whether the attempt SUCCEEDED is not — the bucket means 'attempting entry', not 'entered'."),
    "lateral": StageMapping(
        "lateral", (TA_LATERAL, TA_DISCOVERY),
        "In CIC-IDS2017 this bucket is the Thursday Infiltration episodes (a compromised internal host "
        "scanning the LAN). Held out of training entirely — this is the generalisation split — so a "
        "predicted `lateral` stage on new data rests on the stage head's ability to place unseen behaviour "
        "next to its nearest training bucket, not on having seen lateral movement."),
    "c2": StageMapping(
        "c2", (TA_C2,),
        "Periodic low-volume HTTP beaconing (the Ares botnet on Friday morning). Resolvable from timing "
        "regularity and payload size; the C2 protocol itself is not identified."),
    "exfil": StageMapping(
        "exfil", (TA_IMPACT, TA_EXFIL),
        "IN THIS DATASET the bucket contains only DoS/DDoS episodes — there is no exfiltration in "
        "CIC-IDS2017. A predicted `exfil` stage therefore means Impact-like terminal activity (T1498/T1499); "
        "it is listed under Exfiltration only because the six-bucket taxonomy predates the dataset choice "
        "(labels.py documents the simplification)."),
}

LABEL_MAPPINGS: tuple[LabelMapping, ...] = (
    LabelMapping("PortScan", "Friday", "recon", (T_SCAN_IP, T_NET_SVC_DISCOVERY),
                 "Nmap sweeps from an internal attacker (T1046) — Reconnaissance when the scanner is external."),
    LabelMapping("FTP-Patator", "Tuesday", "initial_access", (T_PASSWORD_GUESS, T_EXTERNAL_REMOTE),
                 "Patator password guessing against FTP."),
    LabelMapping("SSH-Patator", "Tuesday", "initial_access", (T_PASSWORD_GUESS, T_EXTERNAL_REMOTE),
                 "Patator password guessing against SSH."),
    LabelMapping("Web Attack – Brute Force", "Thursday", "initial_access", (T_PASSWORD_GUESS, T_EXPLOIT_PUBLIC),
                 "Login brute force against the DVWA web application.", aliases=("Web Attack - Brute Force",)),
    LabelMapping("Web Attack – XSS", "Thursday", "initial_access", (T_EXPLOIT_PUBLIC,),
                 "Cross-site scripting against DVWA.", aliases=("Web Attack - XSS",)),
    LabelMapping("Web Attack – Sql Injection", "Thursday", "initial_access", (T_EXPLOIT_PUBLIC,),
                 "SQL injection against DVWA.", aliases=("Web Attack - Sql Injection",)),
    LabelMapping("Heartbleed", "Wednesday", "initial_access", (T_EXPLOIT_PUBLIC,),
                 "OpenSSL CVE-2014-0160 memory disclosure; a public-facing exploit whose payload is a credential/"
                 "memory leak, not a foothold."),
    LabelMapping("Infiltration", "Thursday", "lateral", (T_REMOTE_SERVICES, T_NET_SVC_DISCOVERY),
                 "Victim opens a malicious file; the compromised host then scans the internal network. The dataset "
                 "labels the whole episode, so the two sub-phases are not separable per flow."),
    LabelMapping("Bot", "Friday", "c2", (T_WEB_C2,),
                 "Ares botnet HTTP command-and-control."),
    LabelMapping("DoS Hulk", "Wednesday", "exfil", (T_APP_EXHAUST,),
                 "HTTP request flood (application exhaustion)."),
    LabelMapping("DoS GoldenEye", "Wednesday", "exfil", (T_APP_EXHAUST,),
                 "HTTP keep-alive/no-cache flood (application exhaustion)."),
    LabelMapping("DoS slowloris", "Wednesday", "exfil", (T_SVC_EXHAUST,),
                 "Partial HTTP headers holding connections open (service exhaustion)."),
    LabelMapping("DoS Slowhttptest", "Wednesday", "exfil", (T_SVC_EXHAUST,),
                 "Slow-body HTTP (service exhaustion)."),
    LabelMapping("DDoS", "Friday", "exfil", (T_NET_DOS_FLOOD,),
                 "LOIC flood (direct network flood)."),
)

ATTACK_LABELS: tuple[str, ...] = tuple(m.label for m in LABEL_MAPPINGS)


def label_mapping(raw_label: str) -> LabelMapping | None:
    """Exact (case-insensitive, dash-normalised) lookup of a CIC-IDS2017 label."""
    key = _norm(raw_label)
    for m in LABEL_MAPPINGS:
        if key == _norm(m.label) or any(key == _norm(a) for a in m.aliases):
            return m
    return None


def _norm(s: str) -> str:
    return " ".join(str(s).replace("–", "-").replace("—", "-").lower().split())


def stage_tactics(stage: str) -> list[dict]:
    """Tactics for one stage bucket, as JSON-ready dicts."""
    if stage not in STAGE_MAPPINGS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_LABELS}")
    return [{"id": t.id, "name": t.name} for t in STAGE_MAPPINGS[stage].tactics]


def techniques_for_stage(stage: str) -> list[dict]:
    """Every technique any training label in this bucket corresponds to — the
    candidate list the console shows next to a predicted stage."""
    seen: dict[str, Technique] = {}
    for m in LABEL_MAPPINGS:
        if m.stage == stage:
            for t in m.techniques:
                seen.setdefault(t.id, t)
    return [{"id": t.id, "name": t.name, "tactic_id": t.tactic.id} for t in seen.values()]


def map_stage_distribution(stage_probs: dict[str, float], min_prob: float = 0.05) -> list[dict]:
    """Turn one horizon's stage distribution into ranked ATT&CK candidates.
    Benign is skipped; buckets under `min_prob` are skipped; the result is a
    projection of the MODEL's stage output, never a claim about intent."""
    out = []
    for stage, p in sorted(stage_probs.items(), key=lambda kv: -float(kv[1])):
        if stage == "benign" or float(p) < min_prob:
            continue
        out.append({
            "stage": stage,
            "probability": float(p),
            "tactics": stage_tactics(stage),
            "candidate_techniques": techniques_for_stage(stage),
            "resolution": STAGE_MAPPINGS[stage].resolution,
        })
    return out


def progression_summary(horizon_stage_dists: list[dict[str, float]], window_seconds: int) -> dict:
    """Across the K forecast horizons: the argmax non-benign stage per step and
    the first horizon at which each tactic becomes the most likely bucket.
    This is the "attack progression" object the problem statement asks for,
    stated as a sequence of projected buckets with the tactic each maps to."""
    steps = []
    first_seen: dict[str, dict] = {}
    for k, dist in enumerate(horizon_stage_dists, start=1):
        stage = max(dist.items(), key=lambda kv: float(kv[1]))[0]
        p = float(dist[stage])
        tactics = stage_tactics(stage)
        steps.append({"k": k, "t_plus_s": k * window_seconds, "stage": stage, "probability": p, "tactics": tactics})
        for t in tactics:
            first_seen.setdefault(t["id"], {"tactic": t, "first_k": k, "first_t_plus_s": k * window_seconds, "stage": stage})
    return {"steps": steps, "tactics_in_order": sorted(first_seen.values(), key=lambda d: d["first_k"]),
            "attack_version": ATTACK_VERSION, "label": "projected stage sequence (model-internal)"}


def validate_tables() -> None:
    """Every label must map to the stage labels.py assigns it, and every
    stage bucket must be covered by the taxonomy."""
    for m in LABEL_MAPPINGS:
        via_rules = map_label_to_stage(m.label)
        if via_rules != m.stage:
            raise AssertionError(f"{m.label!r}: attack_mapping says {m.stage}, labels.py says {via_rules}")
        if m.stage not in STAGE_LABELS:
            raise AssertionError(f"{m.label!r}: unknown stage {m.stage}")
    if set(STAGE_MAPPINGS) != set(STAGE_LABELS):
        raise AssertionError(f"stage mappings {sorted(STAGE_MAPPINGS)} != STAGE_LABELS {STAGE_LABELS}")


def mapping_markdown() -> str:
    lines = [
        "# NIDRA → MITRE ATT&CK mapping",
        "",
        f"Generated from `nidra/data/attack_mapping.py` ({ATTACK_VERSION}). Do not edit by hand;",
        "run `python -m nidra.data.attack_mapping --write`.",
        "",
        "Both tables are curated presentation mappings. The model predicts a **stage bucket**; the",
        "tactic is the bucket's documented meaning, and the techniques are those of the tools the",
        "dataset authors ran. Nothing here is inferred from traffic beyond the bucket itself.",
        "",
        "## Stage bucket → tactic (what a predicted stage means)",
        "",
        "| Stage | Tactics | What network telemetry can and cannot resolve |",
        "|---|---|---|",
    ]
    for stage in STAGE_LABELS:
        m = STAGE_MAPPINGS[stage]
        tactics = ", ".join(f"{t.id} {t.name}" for t in m.tactics) or "—"
        lines.append(f"| `{stage}` | {tactics} | {m.resolution} |")
    lines += [
        "",
        "## CIC-IDS2017 label → stage → technique (training/evaluation ground truth)",
        "",
        "| Label | Day | Stage | Technique(s) | Note |",
        "|---|---|---|---|---|",
    ]
    for m in LABEL_MAPPINGS:
        techs = "; ".join(f"{t.id} {t.name} ({t.tactic.id})" for t in m.techniques)
        lines.append(f"| {m.label} | {m.day} | `{m.stage}` | {techs} | {m.note} |")
    lines += [
        "",
        "## Coverage and limits",
        "",
        "- Thursday (Web Attacks, Infiltration) is **held out of training entirely**; its rows appear",
        "  here because they are evaluation ground truth for the generalisation split.",
        "- There is no exfiltration, no persistence, no privilege escalation and no defence evasion",
        "  in CIC-IDS2017. Those tactics are not in the taxonomy and the model cannot predict them.",
        "- Technique resolution is by label, not by model: a predicted `initial_access` bucket lists",
        "  every technique any `initial_access` training label carried. The console shows them as",
        "  candidates, not as a classification.",
        "- A projected stage sequence is a **model-internal** object (what the learned dynamics",
        "  roll the host into). It is not attribution and not intent.",
        "",
    ]
    return "\n".join(lines)


def _doc_path() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "ATTACK_MAPPING.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write docs/ATTACK_MAPPING.md")
    args = parser.parse_args()
    validate_tables()
    md = mapping_markdown()
    if args.write:
        _doc_path().parent.mkdir(parents=True, exist_ok=True)
        _doc_path().write_text(md)
        print(f"wrote {_doc_path()}")
    else:
        print(md)


if __name__ == "__main__":
    main()
