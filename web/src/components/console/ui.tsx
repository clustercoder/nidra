"use client";

import * as React from "react";
import { X } from "lucide-react";
import type { LucideIcon } from "lucide-react";

/* Shared console chrome. The dashboard reads as one system only if the chip,
   the pill and the tile are defined once — every panel that rolls its own is a
   place where the spacing and the radius drift. */

export type Tone = "neutral" | "observed" | "projected" | "alert" | "good" | "bad";

const TONE_FG: Record<Tone, string> = {
  neutral: "text-console-muted",
  observed: "text-observed-lit",
  projected: "text-projected-lit",
  alert: "text-threshold-lit",
  good: "text-positive-lit",
  bad: "text-negative-lit",
};

const TONE_BG: Record<Tone, string> = {
  neutral: "bg-console-muted/12",
  observed: "bg-observed-lit/12",
  projected: "bg-projected-lit/12",
  alert: "bg-threshold-lit/12",
  good: "bg-positive-lit/12",
  bad: "bg-negative-lit/12",
};

export function IconChip({
  Icon,
  tone = "neutral",
  size = "md",
}: {
  Icon: LucideIcon;
  tone?: Tone;
  size?: "sm" | "md";
}) {
  const box = size === "sm" ? "size-7 rounded-lg" : "size-9 rounded-xl";
  const glyph = size === "sm" ? "size-3.5" : "size-4";
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center ${box} ${TONE_BG[tone]} ${TONE_FG[tone]}`}
      aria-hidden
    >
      <Icon className={glyph} strokeWidth={2} />
    </span>
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
  format = (v) => String(v),
}: {
  options: readonly T[];
  value: T;
  onChange: (v: T) => void;
  label: string;
  format?: (v: T) => string;
}) {
  return (
    <div
      role="group"
      aria-label={label}
      className="bg-console-raised/70 border-console-line inline-flex items-center gap-0.5 rounded-full border p-0.5"
    >
      {options.map((opt) => {
        const active = opt === value;
        return (
          <button
            key={String(opt)}
            onClick={() => onChange(opt)}
            aria-pressed={active}
            className={`rounded-full px-2.5 py-1 text-[11px] font-medium tabular-nums transition-colors ${
              active
                ? "bg-console-surface text-console-text shadow-sm"
                : "text-console-muted hover:text-console-text"
            }`}
          >
            {format(opt)}
          </button>
        );
      })}
    </div>
  );
}

/** A scope selector in the toolbar: "Host: All ⌄". Native select keeps the
 * keyboard and screen-reader behaviour without a popover implementation. */
export function ScopePill({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="border-console-line bg-console-surface hover:border-console-muted/40 group relative inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors">
      <span className="text-console-muted">{label}:</span>
      <span className="text-console-text font-medium">
        {options.find((o) => o.value === value)?.label ?? value}
      </span>
      <svg viewBox="0 0 12 12" className="text-console-muted size-3" aria-hidden>
        <path d="M3 4.5 6 8l3-3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        className="absolute inset-0 cursor-pointer opacity-0"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function FilterChip({
  label,
  value,
  onClear,
}: {
  label: string;
  value: string;
  onClear: () => void;
}) {
  return (
    <span className="border-observed-lit/40 bg-observed-lit/10 text-console-text inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px]">
      <span className="text-console-muted">{label}:</span>
      <span className="font-medium">{value}</span>
      <button
        onClick={onClear}
        aria-label={`Clear ${label} filter`}
        className="text-console-muted hover:text-console-text transition-colors"
      >
        <X className="size-3" />
      </button>
    </span>
  );
}

export function Sparkline({
  series,
  tone = "observed",
  className = "h-6 w-20",
}: {
  series: number[];
  tone?: Tone;
  className?: string;
}) {
  const id = React.useId();
  if (series.length < 2) return <span className={`block ${className}`} />;
  const w = 80;
  const h = 24;
  const pad = 2;
  /* Sparklines scale to their own range, unlike the main chart: at fleet-wide
     risk of 0.06 a [0,1] sparkline is a flat line that says nothing, and the
     number beside it already carries the absolute value. The floor on the span
     keeps that from running the other way — without it a 0.003 wobble fills
     the box and reads as a spike. */
  const min = Math.min(...series);
  const max = Math.max(...series);
  const span = Math.max(max - min, 0.08);
  const pt = (v: number, i: number) => {
    const x = pad + (i / (series.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / span) * (h - pad * 2);
    return [x, y] as const;
  };
  const line = series.map((v, i) => {
    const [x, y] = pt(v, i);
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  });
  const [lastX] = pt(series[series.length - 1], series.length - 1);
  const [firstX] = pt(series[0], 0);
  const area = [...line, `L${lastX.toFixed(1)} ${h}`, `L${firstX.toFixed(1)} ${h}`, "Z"].join(" ");

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className={`${className} shrink-0`} preserveAspectRatio="none" aria-hidden>
      <defs>
        <linearGradient id={`spark-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.35" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      <g className={TONE_FG[tone]}>
        <path d={area} fill={`url(#spark-${id})`} />
        <path
          d={line.join(" ")}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      </g>
    </svg>
  );
}

/** Single-value tile: icon chip, big number, and a delta line that says what
 * the number means rather than restating it. */
export function StatTile({
  Icon,
  label,
  value,
  unit,
  note,
  noteTone = "neutral",
  tone = "neutral",
  series,
}: {
  Icon: LucideIcon;
  label: string;
  value: string;
  unit?: string;
  note?: string;
  noteTone?: Tone;
  tone?: Tone;
  series?: number[];
}) {
  return (
    <div className="border-console-line bg-console-surface flex flex-col gap-3 rounded-2xl border p-4">
      <div className="flex items-center gap-2.5">
        <IconChip Icon={Icon} tone={tone} size="sm" />
        <span className="text-console-muted text-[11px] font-medium tracking-wide">{label}</span>
      </div>
      <div className="flex items-end justify-between gap-3">
        <div className="flex items-baseline gap-1.5">
          <span
            className={`font-headings text-[30px] leading-none font-semibold tabular-nums ${
              tone === "alert" ? "text-threshold-lit" : "text-console-text"
            }`}
          >
            {value}
          </span>
          {unit && <span className="text-console-muted text-xs">{unit}</span>}
        </div>
        {series && series.length > 1 && (
          <Sparkline series={series} tone={tone === "neutral" ? "observed" : tone} className="h-6 w-16" />
        )}
      </div>
      {note && (
        <p className="flex items-center gap-1.5 text-[11px]">
          <span className={`size-1.5 shrink-0 rounded-full ${TONE_BG[noteTone].replace("/12", "")}`} />
          <span className={TONE_FG[noteTone]}>{note}</span>
        </p>
      )}
    </div>
  );
}

export { TONE_FG, TONE_BG };
