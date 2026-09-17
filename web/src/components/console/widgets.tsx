"use client";

import { Pause, Play } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type {
  ExplainInfo,
  HorizonPoint,
  SignalAttribution,
} from "@/lib/demo-replay";
import { STAGES, clockOf } from "@/lib/demo-replay";
import { SeverityBadge, stageBarClass } from "./severity";
import { IconChip, Segmented, type Tone } from "./ui";

/* ── shell ───────────────────────────────────────────────────────────── */

export function Panel({
  title,
  subtitle,
  Icon,
  iconTone = "neutral",
  aside,
  children,
  className = "",
  bodyClass = "p-4",
}: {
  title?: string;
  subtitle?: string;
  Icon?: LucideIcon;
  iconTone?: Tone;
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section
      className={`border-console-line bg-console-surface flex flex-col overflow-hidden rounded-2xl border ${className}`}
    >
      {title && (
        <header className="flex shrink-0 items-center gap-3 px-4 pt-3.5 pb-3">
          {Icon && <IconChip Icon={Icon} tone={iconTone} size="sm" />}
          <div className="min-w-0">
            <h2 className="text-console-text text-[13px] font-medium">{title}</h2>
            {subtitle && <p className="text-console-muted mt-0.5 text-[11px]">{subtitle}</p>}
          </div>
          {aside && <div className="ml-auto shrink-0">{aside}</div>}
        </header>
      )}
      <div className={`flex-1 ${bodyClass}`}>{children}</div>
    </section>
  );
}

/* ── the alert line ───────────────────────────────────────────────────── */

export function AlertStrip({
  host,
  leadSeconds,
  stage,
  risk,
}: {
  host: string;
  leadSeconds: number;
  stage: string;
  risk: number;
}) {
  return (
    <div className="border-threshold-lit/35 bg-threshold-lit/[0.08] flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border px-4 py-3">
      <span className="relative flex size-2.5 shrink-0">
        <span className="bg-threshold-lit absolute inline-flex size-full animate-ping rounded-full opacity-60" />
        <span className="bg-threshold-lit relative inline-flex size-2.5 rounded-full" />
      </span>
      <span className="text-console-text text-sm font-semibold whitespace-nowrap">
        Threshold crossing forecast
      </span>
      <SeverityBadge stage={stage} />
      <span className="text-console-muted font-mono text-sm">{host}</span>
      <span className="text-console-muted text-sm">
        heading for <span className="text-threshold-lit font-semibold">{stage}</span>
      </span>
      <span className="text-console-muted text-sm">
        risk{" "}
        <span className="text-threshold-lit font-mono font-semibold tabular-nums">
          {risk.toFixed(3)}
        </span>
      </span>
      <span className="ml-auto flex items-baseline gap-1.5">
        <span className="text-threshold-lit font-headings text-2xl leading-none font-semibold tabular-nums">
          {leadSeconds}
        </span>
        <span className="text-console-muted text-sm">seconds of warning</span>
      </span>
    </div>
  );
}

/* ── where a host sits on the lifecycle ───────────────────────────────── */

export function LifecycleTrack({
  observedStage,
  predicted,
}: {
  observedStage: string;
  predicted: HorizonPoint;
}) {
  const nowIdx = STAGES.indexOf(observedStage as (typeof STAGES)[number]);
  const top = STAGES.reduce(
    (best, s) =>
      (predicted.stage_dist[s] ?? 0) > (predicted.stage_dist[best] ?? 0) ? s : best,
    STAGES[0],
  );
  const topIdx = STAGES.indexOf(top);

  return (
    <div>
      <ol className="flex items-stretch gap-1">
        {STAGES.map((stage, i) => {
          const reached = i <= nowIdx;
          const projected = i > nowIdx && i <= topIdx;
          const share = predicted.stage_dist[stage] ?? 0;
          return (
            <li key={stage} className="min-w-0 flex-1">
              <div
                className={`h-1.5 rounded-full ${
                  reached
                    ? stageBarClass(stage)
                    : projected
                      ? "bg-threshold-lit/45"
                      : "bg-console-line"
                }`}
              />
              <div
                className={`mt-2 truncate text-[11px] ${
                  i === nowIdx
                    ? "text-console-text font-semibold"
                    : projected
                      ? "text-threshold-lit"
                      : "text-console-muted"
                }`}
                title={stage}
              >
                {stage}
              </div>
              <div className="text-console-muted mt-0.5 font-mono text-[10px] tabular-nums">
                {(share * 100).toFixed(0)}%
              </div>
            </li>
          );
        })}
      </ol>
      <p className="text-console-muted mt-3 text-xs">
        Observed <span className="text-console-text">{observedStage}</span>.
        At k={predicted.k} the mass sits on{" "}
        <span className="text-threshold-lit">{top}</span> (
        {((predicted.stage_dist[top] ?? 0) * 100).toFixed(1)}%).
      </p>
    </div>
  );
}

/* ── drivers ──────────────────────────────────────────────────────────── */

export function Drivers({
  signals,
  explain,
}: {
  signals: SignalAttribution[];
  explain?: ExplainInfo;
}) {
  const max = Math.max(...signals.map((s) => Math.abs(s.shap_value)), 0.0001);
  return (
    <div>
      {/* A signed contribution is unreadable without knowing which way is bad.
          Positive pushes risk up, so positive is the alarm colour — the earlier
          green-for-positive read as "this is fine" about the thing driving the
          host toward compromise. */}
      <div className="text-console-muted mb-3 flex items-center justify-between text-[10px] tracking-wide uppercase">
        <span className="text-positive-lit">← lowers risk</span>
        <span className="text-negative-lit">raises risk →</span>
      </div>
      <ul className="space-y-3">
        {signals.slice(0, 5).map((s) => {
          const pos = s.shap_value >= 0;
          return (
            <li key={s.name}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-console-text truncate text-[13px]">{s.display}</span>
                <span
                  className={`shrink-0 font-mono text-xs tabular-nums ${
                    pos ? "text-negative-lit" : "text-positive-lit"
                  }`}
                >
                  {pos ? "+" : ""}
                  {s.shap_value.toFixed(3)}
                </span>
              </div>
              <div className="bg-console-line mt-1.5 flex h-1.5 overflow-hidden rounded-full">
                <span className="flex w-1/2 justify-end">
                  {!pos && (
                    <span
                      className="bg-positive-lit block h-full rounded-full"
                      style={{ width: `${(Math.abs(s.shap_value) / max) * 100}%` }}
                    />
                  )}
                </span>
                <span className="flex w-1/2">
                  {pos && (
                    <span
                      className="bg-negative-lit block h-full rounded-full"
                      style={{ width: `${(s.shap_value / max) * 100}%` }}
                    />
                  )}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
      {explain && (
        <div className="border-console-line mt-4 border-t pt-3">
          {/* Early in the replay the context holds one or two windows, and a
              two-bar chart of a one-window history is a slab, not a reading —
              the sentence carries it until there is a shape to draw. */}
          {explain.window_importance.length >= 3 && (
            <div className="flex h-8 items-end gap-px" aria-hidden>
              {explain.window_importance.map((v, i) => {
                const peak = Math.max(...explain.window_importance, 0.0001);
                return (
                  <span
                    key={i}
                    className={`flex-1 self-end rounded-t-[2px] ${
                      i === explain.driving_window ? "bg-projected-lit" : "bg-projected-lit/30"
                    }`}
                    style={{ height: `${Math.max(3, (v / peak) * 32)}px` }}
                  />
                );
              })}
            </div>
          )}
          <p className="text-console-muted mt-2 text-[11px]">
            Saliency across {explain.context_windows} context{" "}
            {explain.context_windows === 1 ? "window" : "windows"} (L={explain.context_l}). Window{" "}
            {explain.driving_window} drove it.
          </p>
        </div>
      )}
    </div>
  );
}

/* ── replay ───────────────────────────────────────────────────────────── */

export function ReplayBar({
  windowCount,
  windowDeltaS,
  t,
  playing,
  speed,
  speeds,
  label,
  onSeek,
  onPlayingChange,
  onSpeedChange,
}: {
  windowCount: number;
  windowDeltaS: number;
  t: number;
  playing: boolean;
  speed: number;
  speeds: readonly number[];
  label: string;
  onSeek: (t: number) => void;
  onPlayingChange: (p: boolean) => void;
  onSpeedChange: (s: number) => void;
}) {
  return (
    <div className="border-console-line bg-console-surface flex flex-wrap items-center gap-x-4 gap-y-3 rounded-2xl border px-4 py-3">
      <button
        onClick={() => onPlayingChange(!playing)}
        aria-label={playing ? "Pause replay" : "Play replay"}
        className="bg-observed-lit text-console-bg flex size-9 shrink-0 items-center justify-center rounded-full transition-transform hover:scale-105"
      >
        {playing ? (
          <Pause className="size-4 fill-current" />
        ) : (
          <Play className="size-4 translate-x-px fill-current" />
        )}
      </button>
      <span className="text-console-text w-[88px] shrink-0 font-mono text-sm tabular-nums">
        {label}
      </span>
      <input
        type="range"
        min={0}
        max={Math.max(0, windowCount - 1)}
        value={t}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="Replay position"
        className="accent-observed-lit min-w-[180px] flex-1"
      />
      <span className="text-console-muted shrink-0 font-mono text-xs tabular-nums">
        {t + 1}/{windowCount} · {windowDeltaS}s
      </span>
      <Segmented
        label="Replay speed"
        options={speeds}
        value={speed}
        onChange={onSpeedChange}
        format={(s) => `${s}×`}
      />
    </div>
  );
}

/* ── stage mix ────────────────────────────────────────────────────────── */

export function StageMix({
  horizons,
  selectedK,
  onSelectK,
  windowDeltaS,
}: {
  horizons: HorizonPoint[];
  selectedK: number;
  onSelectK: (k: number) => void;
  windowDeltaS: number;
}) {
  const point = horizons.find((h) => h.k === selectedK) ?? horizons[0];
  return (
    <div>
      <div className="mb-3">
        <Segmented
          label="Horizon step"
          options={horizons.map((h) => h.k)}
          value={point.k}
          onChange={onSelectK}
          format={(k) => `+${k * windowDeltaS}s`}
        />
      </div>
      <p className="text-console-muted text-xs">
        {clockOf(point.ts)} UTC · risk{" "}
        <span className="text-console-text font-mono tabular-nums">
          {point.p_compromise.toFixed(3)}
        </span>{" "}
        [{point.ci_low.toFixed(2)}–{point.ci_high.toFixed(2)}]
      </p>
      <ul className="mt-3 space-y-2">
        {STAGES.map((stage) => {
          const v = point.stage_dist[stage] ?? 0;
          return (
            <li key={stage} className="flex items-center gap-3 text-xs">
              <span className="text-console-muted w-[88px] shrink-0 truncate">
                {stage}
              </span>
              <span className="bg-console-line h-1.5 flex-1 overflow-hidden rounded-full">
                <span
                  className={`block h-full rounded-full ${stageBarClass(stage)}`}
                  style={{ width: `${v * 100}%` }}
                />
              </span>
              <span className="text-console-muted w-10 shrink-0 text-right font-mono tabular-nums">
                {(v * 100).toFixed(0)}%
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
