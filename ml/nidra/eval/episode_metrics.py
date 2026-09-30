"""Per-episode behaviour of a score: pre-onset warning, detection latency,
lead time — computed on the evaluation set's rows, which contain every
window of every attack episode and every window in the 30 minutes before
each onset (eval_set.py keeps those strata uncapped).

Definitions (all in windows of `window_seconds`):
  warned pre-onset   `m` consecutive windows with score >= threshold inside
                     the pre-onset span (default 30 min) and strictly before
                     the onset
  lead time          onset_ts - first window of that run (seconds), the
                     project's existing lead-time definition
  detected           `m` consecutive windows >= threshold at or after the
                     onset and before the episode ends
  latency            minutes from onset to the first window of that run
"""

from __future__ import annotations

import numpy as np

from nidra.eval.eval_set import EvalSet


def _first_run(hit: np.ndarray, m: int) -> int | None:
    run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        if run >= m:
            return i - m + 1
    return None


def per_episode_report(ev: EvalSet, score: np.ndarray, threshold: float, m: int = 2,
                       pre_onset_minutes: int = 30) -> dict:
    hosts = ev.arrays.host_id
    ts = ev.arrays.origin_ts
    keys = ev.episode_key
    episodes = sorted({k for k in keys if k})
    rows = []
    for key in episodes:
        host, start = key.rsplit("@", 1)
        start = int(start)
        mask = (keys == key) & (hosts == host)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        order = np.argsort(ts[idx])
        idx = idx[order]
        t = ts[idx]
        s = score[idx]
        inside = ev.inside_episode[idx]
        end = int(t[inside].max()) if inside.any() else start
        pre = (t < start) & (t >= start - pre_onset_minutes * 60)
        dur = inside
        hit = s >= threshold
        pre_idx = np.where(pre)[0]
        dur_idx = np.where(dur)[0]
        pw = _first_run(hit[pre_idx], m) if len(pre_idx) else None
        dw = _first_run(hit[dur_idx], m) if len(dur_idx) else None
        rows.append({
            "episode": key, "host": host, "onset_ts": start, "end_ts": end,
            "length_windows": int(dur.sum()),
            "n_pre_onset_rows": int(pre.sum()),
            "warned_pre_onset": pw is not None,
            "lead_time_s": float(start - t[pre_idx[pw]]) if pw is not None else None,
            "max_score_pre_onset": float(s[pre].max()) if pre.any() else None,
            "detected_within_episode": dw is not None,
            "latency_min": float((t[dur_idx[dw]] - start) / 60.0) if dw is not None else None,
            "max_score_inside": float(s[dur].max()) if dur.any() else None,
            "fraction_inside_above_threshold": float(hit[dur].mean()) if dur.any() else None,
        })
    lead = [r["lead_time_s"] for r in rows if r["lead_time_s"] is not None]
    lat = [r["latency_min"] for r in rows if r["latency_min"] is not None]
    return {
        "threshold": float(threshold), "m_consecutive": int(m), "pre_onset_minutes": pre_onset_minutes,
        "n_episodes": len(rows),
        "n_warned_pre_onset": int(sum(r["warned_pre_onset"] for r in rows)),
        "n_detected_within_episode": int(sum(r["detected_within_episode"] for r in rows)),
        "median_lead_time_s": float(np.median(lead)) if lead else None,
        "lead_time_distribution_s": lead,
        "median_latency_min": float(np.median(lat)) if lat else None,
        "latency_distribution_min": lat,
        "episodes": rows,
    }
