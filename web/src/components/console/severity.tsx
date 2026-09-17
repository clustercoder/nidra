/**
 * Stage → severity ramp, the same five-step scale Splunk ES uses for notable
 * events (info/low/medium/high/critical). Kept separate from the chart's
 * observed/projected roles in forecast-chart.tsx — those encode "which line
 * is this", this encodes "how bad is this stage" — so the two must not be
 * unified even where a colour looks reusable.
 */
export const SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"] as const;
export type Severity = (typeof SEVERITY_ORDER)[number];

export const STAGE_SEVERITY: Record<string, Severity> = {
  benign: "info",
  recon: "low",
  initial_access: "medium",
  lateral: "high",
  c2: "critical",
  exfil: "critical",
};

const SEVERITY_STYLE: Record<Severity, { dot: string; badge: string; text: string }> = {
  info: {
    dot: "bg-console-muted",
    badge: "bg-console-muted/15 text-console-muted",
    text: "text-console-muted",
  },
  low: {
    dot: "bg-observed-lit",
    badge: "bg-observed-lit/15 text-observed-lit",
    text: "text-observed-lit",
  },
  medium: {
    dot: "bg-amber-400",
    badge: "bg-amber-400/15 text-amber-400",
    text: "text-amber-400",
  },
  high: {
    dot: "bg-threshold-lit",
    badge: "bg-threshold-lit/15 text-threshold-lit",
    text: "text-threshold-lit",
  },
  critical: {
    dot: "bg-negative-lit",
    badge: "bg-negative-lit/15 text-negative-lit",
    text: "text-negative-lit",
  },
};

export function severityOf(stage: string): Severity {
  return STAGE_SEVERITY[stage] ?? "info";
}

export function SeverityBadge({
  stage,
  className = "",
}: {
  stage: string;
  className?: string;
}) {
  const sev = severityOf(stage);
  const style = SEVERITY_STYLE[sev];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${style.badge} ${className}`}
    >
      <span className={`size-1.5 shrink-0 rounded-full ${style.dot}`} />
      {sev}
    </span>
  );
}

export function SeverityDot({ stage, pulse = false }: { stage: string; pulse?: boolean }) {
  const style = SEVERITY_STYLE[severityOf(stage)];
  return (
    <span className="relative flex size-2 shrink-0">
      {pulse && (
        <span className={`absolute inline-flex size-full animate-ping rounded-full opacity-60 ${style.dot}`} />
      )}
      <span className={`relative inline-flex size-2 rounded-full ${style.dot}`} />
    </span>
  );
}

export function severityRank(stage: string): number {
  return SEVERITY_ORDER.indexOf(severityOf(stage));
}

/** Solid bg-* class for the same ramp, for bars/tracks rather than badges —
 * kept as one function so a stage never disagrees with itself across the
 * lifecycle track, stage mix and the event tables. */
export function stageBarClass(stage: string): string {
  return SEVERITY_STYLE[severityOf(stage)].dot;
}
