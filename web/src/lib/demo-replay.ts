/**
 * The committed replay fixture, and the shapes the console renders.
 *
 * This module is the whole data surface for the committed replay. It reads one
 * JSON file and nothing else — no network, no API client — so /demo renders
 * with every container stopped. The fixture is not synthesised: every value in
 * it is the trained ensemble's own output on real CIC-IDS2017 windows (see
 * `source.summary` and scripts/make_demo_replay.py).
 *
 * `UploadedReplay` is the same shape with labels removed and fidelity notes
 * added — what /api/analyze-pcap returns for a capture the model has never
 * seen. The console renders both through the same components, so there is no
 * second rendering path that could present an upload more confidently than the
 * evidence supports.
 */

import fixture from "@/fixtures/demo-replay.json";

export const STAGES = [
  "benign",
  "recon",
  "initial_access",
  "lateral",
  "c2",
  "exfil",
] as const;

export type Stage = (typeof STAGES)[number];

export type SignalAttribution = {
  name: string;
  /** Signed — the sign is the direction of contribution. */
  shap_value: number;
  direction: "up" | "down";
  display: string;
};

export type HorizonPoint = {
  /** 1..horizon_K */
  k: number;
  ts: string;
  p_compromise: number;
  /** 0 <= ci_low <= p_compromise <= ci_high <= 1 */
  ci_low: number;
  ci_high: number;
  stage_dist: Record<string, number>;
};

/** One host, one origin, K steps forward. The unit the console renders. */
export type Forecast = {
  host_id: string;
  /** Time t. Nothing after this was observed when the forecast was made. */
  origin_ts: string;
  horizons: HorizonPoint[];
  /** null means the threshold is never crossed inside the horizon. */
  lead_time_s: number | null;
  observed_stage: string;
  /** Risk at origin_ts — the cone's anchor. */
  observed_risk: number;
  top_signals: SignalAttribution[];
  driving_window: number;
  model_version: string;
};

export type ModelConfig = {
  window_delta: number;
  context_L: number;
  horizon_K: number;
  n_features: number;
  risk_threshold: number;
  lead_time_m: number;
};

export type ModelInfo = {
  impl: string;
  model_version: string;
  schema_ver: string;
  config: ModelConfig;
};

export type ExplainInfo = {
  method: string;
  driving_window: number;
  window_importance: number[];
  context_windows: number;
  context_l: number;
  note?: string;
};

export type ObservedPoint = { ts: string; risk: number };

type Replay = {
  source: { kind: string; generator: string; summary: string };
  model: ModelInfo;
  hosts: string[];
  forecasts: Forecast[];
  explanations: Record<string, ExplainInfo>;
};

/** What the capture upload returns. `fidelity` is measured, not editorial —
 *  see scripts/analyze_pcap.py's FIDELITY_NOTES. */
export type UploadedReplay = Replay & {
  source: Replay["source"] & {
    kind: "upload";
    fidelity: string[];
    capture: {
      filename: string;
      packets: number;
      hosts_in_capture: number;
      hosts_shown: number;
      window_count: number;
      first_window_utc: string;
      last_window_utc: string;
    };
  };
};

export const replay = fixture as unknown as Replay;

/** Forecasts grouped by host, in window order. */
export function forecastsByHost(
  forecasts: Forecast[] = replay.forecasts,
): Record<string, Forecast[]> {
  const out: Record<string, Forecast[]> = {};
  for (const f of forecasts) (out[f.host_id] ??= []).push(f);
  for (const list of Object.values(out))
    list.sort((a, b) => Date.parse(a.origin_ts) - Date.parse(b.origin_ts));
  return out;
}

/**
 * Hosts that carry the full window sequence, and that sequence's length.
 *
 * Ranking hosts against each other at an index is only meaningful when the
 * index is the same instant for all of them, so a host on a different timeline
 * is dropped rather than shown misaligned. Returns null when nothing is left
 * to show — the caller decides whether that is a thrown error (the committed
 * fixture, which is a build-time bug) or a message (an upload, which is data).
 */
export function alignedHosts(
  hosts: string[],
  byHost: Record<string, Forecast[]>,
): { hosts: string[]; windowCount: number } | null {
  const present = hosts.filter((h) => byHost[h]?.length);
  if (present.length === 0) return null;
  const longest = Math.max(...present.map((h) => byHost[h].length));
  const aligned = present.filter((h) => byHost[h].length === longest);
  return { hosts: aligned, windowCount: longest };
}

/**
 * What actually happened over the window this forecast projected into.
 *
 * Each horizon point is matched against the observed risk recorded at that
 * same timestamp later in the replay. Points with no later observation are
 * dropped rather than guessed — and marks that land outside the band are
 * drawn where they fell, because moving them would fabricate the proof.
 */
export function realityOverlay(
  forecast: Forecast,
  history: Forecast[],
): ObservedPoint[] {
  const byOrigin = new Map(
    history.map((f) => [Date.parse(f.origin_ts), f.observed_risk]),
  );
  return forecast.horizons
    .map((h) => ({ ts: h.ts, risk: byOrigin.get(Date.parse(h.ts)) }))
    .filter((p): p is ObservedPoint => p.risk !== undefined);
}

export function clockOf(ts: string): string {
  return new Date(ts).toISOString().slice(11, 19);
}
