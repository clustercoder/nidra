import type { Forecast, ModelConfig } from "@/lib/demo-replay";
import { severityOf, severityRank, type Severity } from "./severity";

/**
 * Plain-English readings of a forecast.
 *
 * The console used to print the same sentence under every row ("no threshold
 * crossing inside horizon", 192 times), which is true and useless. These
 * helpers say what changed instead, so a feed of them reads like an account of
 * the replay rather than a repeated constant.
 */

export type Trend = "rising" | "falling" | "flat";

/** Flat band is ±2% of the threshold — below that the wobble is noise. */
export function trendOf(risk: number, previous: number | undefined, threshold: number): Trend {
  if (previous === undefined) return "flat";
  const delta = risk - previous;
  if (Math.abs(delta) < threshold * 0.02) return "flat";
  return delta > 0 ? "rising" : "falling";
}

export function messageOf(
  forecast: Forecast,
  threshold: number,
  previousRisk?: number,
): string {
  const trend = trendOf(forecast.observed_risk, previousRisk, threshold);
  if (forecast.lead_time_s !== null) {
    return `Projected to cross ${threshold} in ${forecast.lead_time_s}s — stage ${forecast.observed_stage}`;
  }
  if (forecast.observed_risk >= threshold) {
    return `Holding above ${threshold}, no further escalation projected`;
  }
  if (forecast.observed_risk >= threshold * 0.5) {
    return `Elevated and ${trend}, still short of ${threshold}`;
  }
  return `Nominal — risk ${trend === "flat" ? "steady" : trend} across the horizon`;
}

/**
 * Does this window deserve a line in the feed?
 *
 * A feed that prints every host every window says the same sentence four times
 * a tick and teaches the reader to ignore it. Only a state change, a move worth
 * noticing, or anything at the threshold earns a row.
 */
export function isNotable(
  forecast: Forecast,
  previous: Forecast | undefined,
  threshold: number,
): boolean {
  if (forecast.lead_time_s !== null) return true;
  if (forecast.observed_risk >= threshold) return true;
  if (!previous) return true;
  if (severityOf(forecast.observed_stage) !== severityOf(previous.observed_stage)) return true;
  return Math.abs(forecast.observed_risk - previous.observed_risk) >= threshold * 0.02;
}

export type Posture = {
  severity: Severity;
  /** One word for the hero. */
  headline: string;
  /** One sentence under it. */
  detail: string;
  tone: "calm" | "watch" | "alarm";
};

export function postureOf({
  rows,
  geometry,
  crossingHost,
  leadSeconds,
}: {
  rows: { host: string; forecast: Forecast }[];
  geometry: ModelConfig;
  crossingHost?: string;
  leadSeconds?: number | null;
}): Posture {
  const over = rows.filter((r) => r.forecast.observed_risk >= geometry.risk_threshold);
  /* Stage, not just risk. A host sitting in `lateral` under the risk threshold
     is not a nominal fleet, and a posture card that says so would be teaching
     the reader to trust the wrong number. */
  const worstRow = rows.reduce<(typeof rows)[number] | undefined>(
    (acc, r) =>
      !acc || severityRank(r.forecast.observed_stage) > severityRank(acc.forecast.observed_stage)
        ? r
        : acc,
    undefined,
  );
  const worst: Severity = worstRow ? severityOf(worstRow.forecast.observed_stage) : "info";

  if (crossingHost && leadSeconds != null) {
    return {
      severity: "high",
      headline: "Escalating",
      detail: `${crossingHost} is projected to cross ${geometry.risk_threshold} in ${leadSeconds} seconds.`,
      tone: "alarm",
    };
  }
  if (over.length > 0) {
    return {
      severity: worst === "info" ? "medium" : worst,
      headline: "Above threshold",
      detail: `${over.length} of ${rows.length} hosts sit at or above ${geometry.risk_threshold} this window.`,
      tone: "watch",
    };
  }
  if (worstRow && (worst === "high" || worst === "critical" || worst === "medium")) {
    return {
      severity: worst,
      headline: "Elevated",
      detail: `${worstRow.host} is at stage ${worstRow.forecast.observed_stage} with risk ${worstRow.forecast.observed_risk.toFixed(3)}, still under ${geometry.risk_threshold}.`,
      tone: "watch",
    };
  }
  return {
    severity: worst,
    headline: "Nominal",
    detail: `All ${rows.length} hosts are below ${geometry.risk_threshold} with no crossing inside the ${(geometry.horizon_K * geometry.window_delta) / 60}-minute horizon.`,
    tone: "calm",
  };
}

export type Recommendation = {
  id: string;
  tone: "action" | "done";
  title: string;
  body: string;
  actionLabel: string;
};

/** Every recommendation maps to something the console can actually do — a
 * button that only looks like it works is worse than no button. */
export function recommendationsOf({
  rows,
  geometry,
  crossingHost,
  leadSeconds,
  showingReality,
}: {
  rows: { host: string; forecast: Forecast }[];
  geometry: ModelConfig;
  crossingHost?: string;
  leadSeconds?: number | null;
  showingReality: boolean;
}): Recommendation[] {
  const out: Recommendation[] = [];
  const over = rows.filter((r) => r.forecast.observed_risk >= geometry.risk_threshold);

  if (crossingHost && leadSeconds != null) {
    out.push({
      id: "focus-crossing",
      tone: "action",
      title: `Investigate ${crossingHost}`,
      body: `Earliest crossing in the fleet — ${leadSeconds}s of warning before risk reaches ${geometry.risk_threshold}.`,
      actionLabel: "Open host",
    });
  }

  if (over.length > 0) {
    out.push({
      id: "review-over",
      tone: "action",
      title: `Review ${over.length} host${over.length > 1 ? "s" : ""} above threshold`,
      body: `Filtered to windows at or above ${geometry.risk_threshold} across the whole replay.`,
      actionLabel: "Open in Search",
    });
  } else {
    out.push({
      id: "fleet-clear",
      tone: "done",
      title: "Fleet below threshold",
      body: `No host is at or above ${geometry.risk_threshold} in this window.`,
      actionLabel: "Open in Search",
    });
  }

  out.push({
    id: "validate",
    tone: showingReality ? "done" : "action",
    title: showingReality ? "Outcome overlay on" : "Check the forecast against outcome",
    body: showingReality
      ? "Crosses on the chart show the risk actually recorded at each horizon step."
      : "Overlay what actually happened at each horizon step to judge the band.",
    actionLabel: showingReality ? "Hide outcome" : "Show outcome",
  });

  return out;
}
