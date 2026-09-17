"use client";

import * as React from "react";
import { Activity, Clock, Crosshair, Layers, Sparkles, X } from "lucide-react";

import {
  STAGES,
  clockOf,
  type ExplainInfo,
  type Forecast,
  type ModelInfo,
} from "@/lib/demo-replay";
import { messageOf } from "./insight";
import { SeverityBadge, severityOf, stageBarClass } from "./severity";
import { Drivers } from "./widgets";
import { IconChip } from "./ui";

type Tab = "overview" | "signals" | "raw";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "signals", label: "Signals" },
  { key: "raw", label: "Raw data" },
];

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="border-console-line/60 flex items-baseline justify-between gap-4 border-b py-2 last:border-0">
      <dt className="text-console-muted shrink-0 text-xs">{label}</dt>
      <dd className="text-console-text min-w-0 truncate text-right font-mono text-xs tabular-nums">
        {value}
      </dd>
    </div>
  );
}

function Section({
  title,
  children,
  defaultOpen = true,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <section className="border-console-line bg-console-raised/30 rounded-xl border">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-3.5 py-2.5"
      >
        <span className="text-console-text text-[13px] font-medium">{title}</span>
        <span className="text-console-muted text-[11px]">{open ? "Less detail" : "More detail"}</span>
      </button>
      {open && <dl className="px-3.5 pb-2">{children}</dl>}
    </section>
  );
}

function MiniStat({ label, value, tone = "" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="border-console-line bg-console-raised/40 rounded-xl border px-3 py-2.5">
      <p className="text-console-muted text-[10px] tracking-[0.12em] uppercase">{label}</p>
      <p className={`mt-1.5 font-mono text-base tabular-nums ${tone || "text-console-text"}`}>
        {value}
      </p>
    </div>
  );
}

/**
 * Drill-down for one host at one window — the level the dashboard deliberately
 * does not carry. Everything here comes from the same fixture record the chart
 * is drawn from, including the raw JSON, so a reader can check the rendering
 * against the values behind it.
 */
export function DetailDrawer({
  host,
  forecast,
  model,
  explanation,
  threshold,
  previousRisk,
  onClose,
}: {
  host: string;
  forecast: Forecast;
  model: ModelInfo;
  explanation?: ExplainInfo;
  threshold: number;
  previousRisk?: number;
  onClose: () => void;
}) {
  const [tab, setTab] = React.useState<Tab>("overview");

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const geometry = model.config;
  const sev = severityOf(forecast.observed_stage);
  const over = forecast.observed_risk >= threshold;
  const kLast = forecast.horizons[forecast.horizons.length - 1];
  const topStage = STAGES.reduce(
    (best, s) => ((kLast?.stage_dist[s] ?? 0) > (kLast?.stage_dist[best] ?? 0) ? s : best),
    STAGES[0],
  );

  return (
    <>
      <div
        className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[1px]"
        onClick={onClose}
        aria-hidden
      />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`Forecast detail for ${host}`}
        className="border-console-line bg-console-surface fixed inset-y-0 right-0 z-50 flex w-full max-w-[540px] flex-col border-l shadow-2xl"
      >
        <header className="border-console-line shrink-0 border-b px-5 py-4">
          <div className="flex items-start gap-3">
            <IconChip Icon={Crosshair} tone={over ? "alert" : "observed"} />
            <div className="min-w-0 flex-1">
              <h2 className="text-console-text font-mono text-base leading-tight">{host}</h2>
              <p className="text-console-muted mt-1 text-xs">
                Forecast origin {clockOf(forecast.origin_ts)} UTC
              </p>
            </div>
            <button
              onClick={onClose}
              aria-label="Close detail"
              className="text-console-muted hover:bg-console-raised hover:text-console-text rounded-lg p-1.5 transition-colors"
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
            <SeverityBadge stage={forecast.observed_stage} />
            <span className="text-console-muted flex items-center gap-1.5">
              <Layers className="size-3.5" />
              {forecast.observed_stage}
            </span>
            <span className="text-console-muted flex items-center gap-1.5">
              <Clock className="size-3.5" />
              {forecast.lead_time_s === null ? "no crossing" : `${forecast.lead_time_s}s lead`}
            </span>
            <span className="text-console-muted flex items-center gap-1.5">
              <Activity className="size-3.5" />
              {model.model_version}
            </span>
          </div>

          <nav className="border-console-line -mb-4 mt-3 flex gap-4 border-b">
            {TABS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setTab(key)}
                aria-current={tab === key ? "page" : undefined}
                className={`-mb-px border-b-2 pb-2 text-xs font-medium transition-colors ${
                  tab === key
                    ? "border-observed-lit text-console-text"
                    : "text-console-muted hover:text-console-text border-transparent"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </header>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-5">
          {tab === "overview" && (
            <>
              <div className="grid grid-cols-3 gap-2.5">
                <MiniStat
                  label="Risk"
                  value={forecast.observed_risk.toFixed(3)}
                  tone={over ? "text-threshold-lit" : undefined}
                />
                <MiniStat label="Severity" value={sev} />
                <MiniStat
                  label="Lead"
                  value={forecast.lead_time_s === null ? "—" : `${forecast.lead_time_s}s`}
                  tone={forecast.lead_time_s === null ? undefined : "text-threshold-lit"}
                />
              </div>

              <div className="border-console-line bg-console-raised/30 rounded-xl border p-3.5">
                <p className="text-console-text flex items-center gap-2 text-[13px] font-medium">
                  <Sparkles className="text-projected-lit size-3.5" />
                  What this window says
                </p>
                <p className="text-console-muted mt-2 text-xs leading-relaxed">
                  {messageOf(forecast, threshold, previousRisk)}. At the last horizon step
                  (+{(kLast?.k ?? 0) * geometry.window_delta}s) the model puts most stage mass on{" "}
                  <span className="text-console-text">{topStage}</span>, with risk{" "}
                  <span className="text-console-text">{kLast?.p_compromise.toFixed(3)}</span>{" "}
                  inside a {kLast?.ci_low.toFixed(2)}–{kLast?.ci_high.toFixed(2)} band. The band is
                  a calibration measurement, not a guarantee.
                </p>
              </div>

              <Section title="Projected stage mix">
                <ul className="space-y-2 py-1">
                  {STAGES.map((stage) => {
                    const v = kLast?.stage_dist[stage] ?? 0;
                    return (
                      <li key={stage} className="flex items-center gap-3 text-xs">
                        <span className="text-console-muted w-[92px] shrink-0 truncate">{stage}</span>
                        <span className="bg-console-line h-1.5 flex-1 overflow-hidden rounded-full">
                          <span
                            className={`block h-full rounded-full ${stageBarClass(stage)}`}
                            style={{ width: `${v * 100}%` }}
                          />
                        </span>
                        <span className="text-console-muted w-9 shrink-0 text-right font-mono tabular-nums">
                          {(v * 100).toFixed(0)}%
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </Section>

              <Section title="Forecast properties">
                <Row label="Host" value={forecast.host_id} />
                <Row label="Origin" value={`${clockOf(forecast.origin_ts)} UTC`} />
                <Row label="Observed stage" value={forecast.observed_stage} />
                <Row label="Observed risk" value={forecast.observed_risk.toFixed(4)} />
                <Row
                  label="Lead time"
                  value={forecast.lead_time_s === null ? "null" : `${forecast.lead_time_s} s`}
                />
                <Row label="Horizon steps" value={String(forecast.horizons.length)} />
                <Row label="Driving window" value={String(forecast.driving_window)} />
              </Section>

              <Section title="Model & geometry" defaultOpen={false}>
                <Row label="Implementation" value={model.impl} />
                <Row label="Model version" value={forecast.model_version} />
                <Row label="Schema" value={model.schema_ver} />
                <Row label="Window Δ" value={`${geometry.window_delta} s`} />
                <Row label="Context L" value={String(geometry.context_L)} />
                <Row label="Horizon K" value={String(geometry.horizon_K)} />
                <Row label="Features" value={String(geometry.n_features)} />
                <Row label="Risk threshold" value={geometry.risk_threshold.toFixed(2)} />
                <Row label="Lead-time m" value={String(geometry.lead_time_m)} />
              </Section>
            </>
          )}

          {tab === "signals" && (
            <div className="border-console-line bg-console-raised/30 rounded-xl border p-3.5">
              <Drivers signals={forecast.top_signals} explain={explanation} />
            </div>
          )}

          {tab === "raw" && (
            <pre className="border-console-line bg-console-bg text-console-muted overflow-x-auto rounded-xl border p-3.5 font-mono text-[11px] leading-relaxed">
              {JSON.stringify(forecast, null, 2)}
            </pre>
          )}
        </div>
      </aside>
    </>
  );
}
