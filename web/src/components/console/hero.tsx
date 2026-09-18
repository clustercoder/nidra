"use client";

import { AlertTriangle, CheckCircle2, RadioTower, ShieldAlert } from "lucide-react";

import { clockOf } from "@/lib/demo-replay";
import type { Posture } from "./insight";

const TONE = {
  calm: {
    Icon: CheckCircle2,
    ring: "border-positive-lit/25",
    wash: "from-positive-lit/10",
    pill: "bg-positive-lit/15 text-positive-lit",
    dot: "bg-positive-lit",
    glyph: "text-positive-lit",
  },
  watch: {
    Icon: ShieldAlert,
    ring: "border-threshold-lit/30",
    wash: "from-threshold-lit/10",
    pill: "bg-threshold-lit/15 text-threshold-lit",
    dot: "bg-threshold-lit",
    glyph: "text-threshold-lit",
  },
  alarm: {
    Icon: AlertTriangle,
    ring: "border-negative-lit/30",
    wash: "from-negative-lit/12",
    pill: "bg-negative-lit/15 text-negative-lit",
    dot: "bg-negative-lit",
    glyph: "text-negative-lit",
  },
} as const;

/** The posture card: one word for the fleet, one sentence of why, and the
 * controls that act on it. Everything else on the dashboard is detail under
 * this claim. */
export function PostureHero({
  posture,
  originTs,
  hostCount,
  aboveCount,
  threshold,
  onPrimary,
  primaryLabel,
}: {
  posture: Posture;
  originTs: string;
  hostCount: number;
  aboveCount: number;
  threshold: number;
  onPrimary: () => void;
  primaryLabel: string;
}) {
  const tone = TONE[posture.tone];
  const { Icon } = tone;

  return (
    <section
      className={`bg-console-surface relative flex flex-col overflow-hidden rounded-2xl border ${tone.ring} p-5`}
    >
      <div
        className={`pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b ${tone.wash} to-transparent`}
        aria-hidden
      />
      <div className="relative flex flex-col gap-4">
        <div className="flex items-center gap-2 text-[10px] font-semibold tracking-[0.16em] uppercase">
          <span className="text-console-muted">Status</span>
          <span className="text-console-muted/50">·</span>
          <span className="text-console-muted font-mono tracking-normal">
            {clockOf(originTs)} UTC
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Icon className={`size-7 shrink-0 ${tone.glyph}`} strokeWidth={2} />
          <h1 className="font-headings text-console-text text-3xl leading-none font-semibold">
            {posture.headline}
          </h1>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold tracking-wide uppercase ${tone.pill}`}
          >
            <span className={`size-1.5 rounded-full ${tone.dot}`} />
            {posture.severity}
          </span>
        </div>

        <p className="text-console-muted text-sm leading-relaxed">{posture.detail}</p>

        <dl className="border-console-line grid grid-cols-3 gap-3 border-t pt-4">
          {[
            ["Hosts", String(hostCount)],
            ["Above", String(aboveCount)],
            ["Threshold", threshold.toFixed(2)],
          ].map(([k, v]) => (
            <div key={k}>
              <dt className="text-console-muted text-[10px] tracking-[0.12em] uppercase">{k}</dt>
              <dd className="text-console-text mt-1 font-mono text-lg tabular-nums">{v}</dd>
            </div>
          ))}
        </dl>

        <button
          onClick={onPrimary}
          className="border-console-line bg-console-raised text-console-text hover:border-observed-lit/50 hover:text-observed-lit inline-flex w-fit items-center gap-2 rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors"
        >
          <RadioTower className="size-3.5" />
          {primaryLabel}
        </button>
      </div>
    </section>
  );
}
