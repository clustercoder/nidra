"use client";

import { Pause, Play } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type {
  ExplainInfo,
  Forecast,
  HorizonPoint,
  SignalAttribution,
} from "@/lib/demo-replay";
import { STAGES, clockOf } from "@/lib/demo-replay";

/* ── shell ───────────────────────────────────────────────────────────── */

export function Panel({
  title,
  aside,
  children,
  className = "",
  bodyClass = "p-4",
}: {
  title?: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section
      className={`border-console-line bg-console-surface flex flex-col overflow-hidden rounded-xl border ${className}`}
    >
      {title && (
        <header className="border-console-line flex shrink-0 items-center gap-3 border-b px-4 py-2.5">
          <h2 className="text-console-muted text-[11px] font-semibold tracking-[0.14em] uppercase">
            {title}
          </h2>
          {aside && <div className="ml-auto">{aside}</div>}
        </header>
      )}
      <div className={`flex-1 ${bodyClass}`}>{children}</div>
    </section>
  );
}

/* ── the numbers that lead ────────────────────────────────────────────── */

export function Kpi({
  label,
  value,
  unit,
  hint,
  Icon,
  tone = "neutral",
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  Icon: LucideIcon;
  tone?: "neutral" | "alert" | "good";
}) {
  const accent =
    tone === "alert"
      ? "text-threshold-lit"
      : tone === "good"
        ? "text-positive-lit"
        : "text-observed-lit";
  return (
    <div className="border-console-line bg-console-surface relative overflow-hidden rounded-xl border p-4">
      <div className="flex items-center gap-2">
        <Icon className={`size-4 ${accent}`} strokeWidth={2} />
        <span className="text-console-muted text-[11px] font-semibold tracking-[0.12em] uppercase">
          {label}
        </span>
      </div>
      <div className="mt-3 flex items-baseline gap-1.5">
        <span
          className={`font-headings text-[34px] leading-none font-semibold tabular-nums ${
            tone === "alert" ? "text-threshold-lit" : "text-console-text"
          }`}
        >
          {value}
        </span>
        {unit && <span className="text-console-muted text-sm">{unit}</span>}
      </div>
      {hint && <p className="text-console-muted mt-2 text-xs">{hint}</p>}
    </div>
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
    <div className="border-threshold-lit/40 bg-threshold-lit/10 flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border px-4 py-3">
      <span className="relative flex size-2.5 shrink-0">
        <span className="bg-threshold-lit absolute inline-flex size-full animate-ping rounded-full opacity-60" />
        <span className="bg-threshold-lit relative inline-flex size-2.5 rounded-full" />
      </span>
      <span className="text-console-text text-sm font-semibold">
        Threshold crossing forecast
      </span>
      <span className="text-console-muted font-mono text-sm">{host}</span>
      <span className="text-console-muted text-sm">
        heading for{" "}
        <span className="text-threshold-lit font-semibold">{stage}</span>
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

const STAGE_TONE: Record<string, string> = {
  benign: "bg-positive-lit",
  recon: "bg-observed-lit",
  initial_access: "bg-projected-lit",
  lateral: "bg-threshold-lit",
  c2: "bg-negative-lit",
  exfil: "bg-negative-lit",
};

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
                    ? STAGE_TONE[stage]
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

/* ── host watch ───────────────────────────────────────────────────────── */

function Spark({ series, over }: { series: number[]; over: boolean }) {
  if (series.length < 2) return <span className="block h-6" />;
  const w = 64;
  const h = 22;
  const d = series
    .map((v, i) => {
      const x = (i / (series.length - 1)) * w;
      const y = h - Math.max(0, Math.min(1, v)) * h;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-5 w-16 shrink-0" aria-hidden>
      <path
        d={d}
        fill="none"
        className={over ? "stroke-threshold-lit" : "stroke-observed-lit"}
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function HostWatch({
  rows,
  threshold,
  selected,
  onSelect,
}: {
  rows: { host: string; forecast: Forecast; series: number[] }[];
  threshold: number;
  selected: string;
  onSelect: (host: string) => void;
}) {
  return (
    <ul className="divide-console-line divide-y">
      {rows.map(({ host, forecast, series }) => {
        const over = forecast.observed_risk >= threshold;
        const active = host === selected;
        return (
          <li key={host}>
            <button
              onClick={() => onSelect(host)}
              aria-current={active ? "true" : undefined}
              className={`w-full px-4 py-3 text-left transition-colors ${
                active ? "bg-console-raised" : "hover:bg-console-raised/60"
              }`}
            >
              <span className="flex items-center gap-3">
                <span
                  className={`size-2 shrink-0 rounded-full ${over ? "bg-threshold-lit" : "bg-positive-lit/70"}`}
                />
                <span className="text-console-text shrink-0 font-mono text-[13px]">
                  {host}
                </span>
                <span className="flex-1" />
                <Spark series={series} over={over} />
                <span
                  className={`w-12 shrink-0 text-right font-mono text-sm tabular-nums ${
                    over ? "text-threshold-lit font-semibold" : "text-console-muted"
                  }`}
                >
                  {forecast.observed_risk.toFixed(3)}
                </span>
              </span>
              <span className="mt-1.5 flex items-center gap-2 pl-5">
                <span className="text-console-muted text-[11px]">
                  {forecast.observed_stage}
                </span>
                {forecast.lead_time_s !== null && (
                  <span className="bg-threshold-lit/15 text-threshold-lit rounded px-1.5 py-0.5 text-[10px] font-semibold">
                    crossing in {forecast.lead_time_s}s
                  </span>
                )}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
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
      <ul className="space-y-3">
        {signals.slice(0, 5).map((s) => {
          const pos = s.shap_value >= 0;
          return (
            <li key={s.name}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-console-text truncate text-[13px]">
                  {s.display}
                </span>
                <span
                  className={`shrink-0 font-mono text-xs tabular-nums ${pos ? "text-positive-lit" : "text-negative-lit"}`}
                >
                  {pos ? "+" : ""}
                  {s.shap_value.toFixed(3)}
                </span>
              </div>
              <div className="bg-console-line mt-1.5 flex h-1.5 overflow-hidden rounded-full">
                <span className="flex w-1/2 justify-end">
                  {!pos && (
                    <span
                      className="bg-negative-lit block h-full rounded-full"
                      style={{ width: `${(Math.abs(s.shap_value) / max) * 100}%` }}
                    />
                  )}
                </span>
                <span className="flex w-1/2">
                  {pos && (
                    <span
                      className="bg-positive-lit block h-full rounded-full"
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
          <div className="flex items-end gap-px" aria-hidden>
            {explain.window_importance.map((v, i) => {
              const peak = Math.max(...explain.window_importance, 0.0001);
              return (
                <span
                  key={i}
                  className={`h-8 flex-1 self-end rounded-t-[2px] ${
                    i === explain.driving_window
                      ? "bg-projected-lit"
                      : "bg-projected-lit/30"
                  }`}
                  style={{ height: `${Math.max(8, (v / peak) * 32)}px` }}
                />
              );
            })}
          </div>
          <p className="text-console-muted mt-2 text-[11px]">
            Saliency across {explain.context_windows} context windows (L=
            {explain.context_l}). Window {explain.driving_window} drove it.
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
    <div className="border-console-line bg-console-surface flex flex-wrap items-center gap-x-4 gap-y-3 rounded-xl border px-4 py-3">
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
      <div className="flex shrink-0 gap-1" role="group" aria-label="Replay speed">
        {speeds.map((s) => (
          <button
            key={s}
            onClick={() => onSpeedChange(s)}
            aria-pressed={s === speed}
            className={`rounded-md px-2 py-1 font-mono text-xs tabular-nums transition-colors ${
              s === speed
                ? "bg-observed-lit text-console-bg font-semibold"
                : "text-console-muted hover:bg-console-raised"
            }`}
          >
            {s}×
          </button>
        ))}
      </div>
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
      <div className="mb-3 flex items-center gap-1" role="group" aria-label="Horizon step">
        {horizons.map((h) => (
          <button
            key={h.k}
            onClick={() => onSelectK(h.k)}
            aria-pressed={h.k === point.k}
            className={`flex-1 rounded-md py-1.5 font-mono text-xs tabular-nums transition-colors ${
              h.k === point.k
                ? "bg-observed-lit text-console-bg font-semibold"
                : "text-console-muted hover:bg-console-raised"
            }`}
          >
            +{h.k * windowDeltaS}s
          </button>
        ))}
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
                  className={`block h-full rounded-full ${STAGE_TONE[stage]}`}
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
