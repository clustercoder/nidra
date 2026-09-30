# NIDRA → MITRE ATT&CK mapping

Generated from `nidra/data/attack_mapping.py` (Enterprise ATT&CK v14). Do not edit by hand;
run `python -m nidra.data.attack_mapping --write`.

Both tables are curated presentation mappings. The model predicts a **stage bucket**; the
tactic is the bucket's documented meaning, and the techniques are those of the tools the
dataset authors ran. Nothing here is inferred from traffic beyond the bucket itself.

## Stage bucket → tactic (what a predicted stage means)

| Stage | Tactics | What network telemetry can and cannot resolve |
|---|---|---|
| `benign` | — | No tactic. The bucket every host sits in almost all of the time. |
| `recon` | TA0043 Reconnaissance, TA0007 Discovery | Resolvable from flow telemetry: fan-out to many destination ports/hosts with tiny payloads (dst_port_entropy, out_degree, new_peer_count, syn_ratio). External scanning is Reconnaissance; the same behaviour from an internal host is Discovery — the mapping cannot tell them apart without knowing which side of the perimeter the host is on. |
| `initial_access` | TA0001 Initial Access, TA0006 Credential Access | Brute-force logins (many short sessions to one service port) and web exploitation are visible in flow shape; whether the attempt SUCCEEDED is not — the bucket means 'attempting entry', not 'entered'. |
| `lateral` | TA0008 Lateral Movement, TA0007 Discovery | In CIC-IDS2017 this bucket is the Thursday Infiltration episodes (a compromised internal host scanning the LAN). Held out of training entirely — this is the generalisation split — so a predicted `lateral` stage on new data rests on the stage head's ability to place unseen behaviour next to its nearest training bucket, not on having seen lateral movement. |
| `c2` | TA0011 Command and Control | Periodic low-volume HTTP beaconing (the Ares botnet on Friday morning). Resolvable from timing regularity and payload size; the C2 protocol itself is not identified. |
| `exfil` | TA0040 Impact, TA0010 Exfiltration | IN THIS DATASET the bucket contains only DoS/DDoS episodes — there is no exfiltration in CIC-IDS2017. A predicted `exfil` stage therefore means Impact-like terminal activity (T1498/T1499); it is listed under Exfiltration only because the six-bucket taxonomy predates the dataset choice (labels.py documents the simplification). |

## CIC-IDS2017 label → stage → technique (training/evaluation ground truth)

| Label | Day | Stage | Technique(s) | Note |
|---|---|---|---|---|
| PortScan | Friday | `recon` | T1595.001 Active Scanning: Scanning IP Blocks (TA0043); T1046 Network Service Discovery (TA0007) | Nmap sweeps from an internal attacker (T1046) — Reconnaissance when the scanner is external. |
| FTP-Patator | Tuesday | `initial_access` | T1110.001 Brute Force: Password Guessing (TA0006); T1133 External Remote Services (TA0001) | Patator password guessing against FTP. |
| SSH-Patator | Tuesday | `initial_access` | T1110.001 Brute Force: Password Guessing (TA0006); T1133 External Remote Services (TA0001) | Patator password guessing against SSH. |
| Web Attack – Brute Force | Thursday | `initial_access` | T1110.001 Brute Force: Password Guessing (TA0006); T1190 Exploit Public-Facing Application (TA0001) | Login brute force against the DVWA web application. |
| Web Attack – XSS | Thursday | `initial_access` | T1190 Exploit Public-Facing Application (TA0001) | Cross-site scripting against DVWA. |
| Web Attack – Sql Injection | Thursday | `initial_access` | T1190 Exploit Public-Facing Application (TA0001) | SQL injection against DVWA. |
| Heartbleed | Wednesday | `initial_access` | T1190 Exploit Public-Facing Application (TA0001) | OpenSSL CVE-2014-0160 memory disclosure; a public-facing exploit whose payload is a credential/memory leak, not a foothold. |
| Infiltration | Thursday | `lateral` | T1021 Remote Services (TA0008); T1046 Network Service Discovery (TA0007) | Victim opens a malicious file; the compromised host then scans the internal network. The dataset labels the whole episode, so the two sub-phases are not separable per flow. |
| Bot | Friday | `c2` | T1071.001 Application Layer Protocol: Web Protocols (TA0011) | Ares botnet HTTP command-and-control. |
| DoS Hulk | Wednesday | `exfil` | T1499.003 Endpoint Denial of Service: Application Exhaustion Flood (TA0040) | HTTP request flood (application exhaustion). |
| DoS GoldenEye | Wednesday | `exfil` | T1499.003 Endpoint Denial of Service: Application Exhaustion Flood (TA0040) | HTTP keep-alive/no-cache flood (application exhaustion). |
| DoS slowloris | Wednesday | `exfil` | T1499.002 Endpoint Denial of Service: Service Exhaustion Flood (TA0040) | Partial HTTP headers holding connections open (service exhaustion). |
| DoS Slowhttptest | Wednesday | `exfil` | T1499.002 Endpoint Denial of Service: Service Exhaustion Flood (TA0040) | Slow-body HTTP (service exhaustion). |
| DDoS | Friday | `exfil` | T1498.001 Network Denial of Service: Direct Network Flood (TA0040) | LOIC flood (direct network flood). |

## Coverage and limits

- Thursday (Web Attacks, Infiltration) is **held out of training entirely**; its rows appear
  here because they are evaluation ground truth for the generalisation split.
- There is no exfiltration, no persistence, no privilege escalation and no defence evasion
  in CIC-IDS2017. Those tactics are not in the taxonomy and the model cannot predict them.
- Technique resolution is by label, not by model: a predicted `initial_access` bucket lists
  every technique any `initial_access` training label carried. The console shows them as
  candidates, not as a classification.
- A projected stage sequence is a **model-internal** object (what the learned dynamics
  roll the host into). It is not attribution and not intent.
