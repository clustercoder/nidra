/**
 * The committed replay fixture, and the shapes the console renders.
 *
 * This module is the whole data surface for /demo. It reads one JSON file and
 * nothing else — no network, no API client — so the page renders with every
 * container stopped. The fixture is synthesised (see `source.summary`), and the
 * banner on the page says so.
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

export const replay = fixture as unknown as Replay;

/** Forecasts grouped by host, in window order. */
export function forecastsByHost(): Record<string, Forecast[]> {
  const out: Record<string, Forecast[]> = {};
  for (const f of replay.forecasts) (out[f.host_id] ??= []).push(f);
  for (const list of Object.values(out))
    list.sort((a, b) => Date.parse(a.origin_ts) - Date.parse(b.origin_ts));
  return out;
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
