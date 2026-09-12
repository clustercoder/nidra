"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowLeft, Cpu, Radar, ShieldAlert, Timer } from "lucide-react";

import { NidraMark } from "@/components/icons";
import { ConsoleThemeToggle } from "@/components/console/theme-toggle";
import { ForecastChart } from "@/components/console/forecast-chart";
import {
  AlertStrip,
  Drivers,
  HostWatch,
  Kpi,
  LifecycleTrack,
  Panel,
  ReplayBar,
  StageMix,
} from "@/components/console/widgets";
import {
  clockOf,
  realityOverlay,
  type ExplainInfo,
  type Forecast,
  type ModelInfo,
} from "@/lib/demo-replay";

const SPEEDS = [1, 10, 30, 60] as const;
const DEFAULT_SPEED = 60;
const RING_CAP = 200;
const SPARK_WINDOWS = 24;

function useReducedMotion() {
  return React.useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia("(prefers-reduced-motion: reduce)");
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    () => false,
  );
}

export function DemoConsole({
  model,
  hosts,
  byHost,
  explanations,
  sourceSummary,
  initial,
}: {
  model: ModelInfo;
  hosts: string[];
  byHost: Record<string, Forecast[]>;
  explanations: Record<string, ExplainInfo>;
  sourceSummary: string;
  initial: { host?: string; t: number; paused: boolean };
}) {
  const geometry = model.config;
  const windowCount = byHost[hosts[0]]?.length ?? 0;
  const lastIndex = Math.max(0, windowCount - 1);

  const escalating = React.useMemo(
    () =>
      hosts.find((h) => (byHost[h] ?? []).some((f) => f.lead_time_s !== null)) ??
      hosts[0],
    [hosts, byHost],
  );

  const reducedMotion = useReducedMotion();
  const [t, setT] = React.useState(Math.min(initial.t, lastIndex));
  const [speed, setSpeed] = React.useState<number>(DEFAULT_SPEED);
  const [host, setHost] = React.useState(initial.host ?? escalating);
  const [showReality, setShowReality] = React.useState(false);
  const [selectedK, setSelectedK] = React.useState(3);
  const [playingOverride, setPlayingOverride] = React.useState<boolean | null>(
    initial.paused ? false : null,
  );
  const playing = playingOverride ?? !reducedMotion;

  /* One rAF loop drives the clock: wall time scaled by the replay speed and
     converted to whole windows, so 60x advances two windows a second rather
     than re-rendering on every frame. */
  React.useEffect(() => {
    if (!playing || windowCount === 0) return;
    let raf = 0;
    let last = performance.now();
    let carry = 0;
    const tick = (now: number) => {
      carry += (((now - last) / 1000) * speed) / geometry.window_delta;
      last = now;
      if (carry >= 1) {
        const steps = Math.floor(carry);
        carry -= steps;
        setT((prev) => (prev + steps) % windowCount);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, windowCount, geometry.window_delta]);

  React.useEffect(() => {
    const id = setTimeout(() => {
      const q = new URLSearchParams({ host, t: String(t) });
      window.history.replaceState(null, "", `?${q.toString()}`);
    }, 300);
    return () => clearTimeout(id);
  }, [host, t]);

  const riskSeries = React.useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const h of hosts) out[h] = (byHost[h] ?? []).map((f) => f.observed_risk);
    return out;
  }, [hosts, byHost]);

  const hostList = byHost[host] ?? [];
  const current = hostList[Math.min(t, lastIndex)] ?? hostList[0];

  if (!current) {
    return (
      <p className="bg-console-bg text-console-muted min-h-screen p-6">
        The replay fixture is empty — regenerate it with
        scripts/make_demo_fixture.py.
      </p>
    );
  }

  const ti = Math.min(t, lastIndex);
  const rows = hosts
    .map((h) => ({
      host: h,
      forecast: byHost[h]?.[ti],
      series: (riskSeries[h] ?? []).slice(
        Math.max(0, ti + 1 - SPARK_WINDOWS),
        ti + 1,
      ),
    }))
    .filter((r): r is { host: string; forecast: Forecast; series: number[] } =>
      Boolean(r.forecast),
    )
    .sort((a, b) => b.forecast.observed_risk - a.forecast.observed_risk);

  const observed = (hostList ?? [])
    .map((f) => ({ ts: f.origin_ts, risk: f.observed_risk }))
    .slice(Math.max(0, ti + 1 - RING_CAP), ti + 1);

  const reality = showReality ? realityOverlay(current, hostList) : undefined;
  const covered = reality?.filter((p, i) => {
    const h = current.horizons[i];
    return h && p.risk >= h.ci_low && p.risk <= h.ci_high;
  }).length;

  const explanation = explanations[`${host}@${current.origin_ts}`];
  const overCount = rows.filter(
    (r) => r.forecast.observed_risk >= geometry.risk_threshold,
  ).length;
  const crossing = rows
    .filter((r) => r.forecast.lead_time_s !== null)
    .sort(
      (a, b) => (a.forecast.lead_time_s ?? 0) - (b.forecast.lead_time_s ?? 0),
    )[0];
  const kPoint =
    current.horizons.find((h) => h.k === selectedK) ?? current.horizons[0];

  return (
    <div className="bg-console-bg text-console-text min-h-screen">
      <aside
        role="note"
        aria-label="Demo notice"
        title={sourceSummary}
        className="border-threshold-lit/25 bg-threshold-lit/10 text-console-text sticky top-0 z-50 border-b px-4 py-1.5 text-center text-[11px]"
      >
        Demo — synthesised replay shaped on CIC-IDS2017 Wednesday, driven through
        the stub pipeline. No trained model; values are illustrative.
      </aside>

      <header className="border-console-line bg-console-surface/80 flex h-14 shrink-0 items-center gap-3 border-b px-5 backdrop-blur">
        <Link href="/" className="flex items-center gap-2.5">
          <NidraMark className="text-observed-lit size-7" />
          <span className="font-headings text-console-text text-base font-bold tracking-wide">
            NIDRA
          </span>
        </Link>
        <span className="text-console-muted border-console-line ml-1 border-l pl-3 text-sm">
          demo console
        </span>
        <span className="bg-positive-lit/15 text-positive-lit ml-2 hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium sm:inline-flex">
          <span className="bg-positive-lit size-1.5 rounded-full" />
          replay live
        </span>
        <div className="text-console-muted ml-auto flex items-center gap-4 text-xs">
          <span className="hidden font-mono md:inline">
            {model.impl} · {model.model_version}
          </span>
          <ConsoleThemeToggle />
          <Link
            href="/"
            className="hover:text-console-text inline-flex items-center gap-1.5 transition-colors"
          >
            <ArrowLeft className="size-4" />
            Back to site
          </Link>
        </div>
      </header>

      <main className="space-y-4 p-4">
        {/* the numbers first — the chart is one panel among several */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Kpi
            Icon={Timer}
            label="Earliest warning"
            value={crossing ? String(crossing.forecast.lead_time_s) : "—"}
            unit={crossing ? "s" : undefined}
            tone={crossing ? "alert" : "good"}
            hint={
              crossing
                ? `${crossing.host} crosses ${geometry.risk_threshold} first`
                : "No host projected to cross the threshold"
            }
          />
          <Kpi
            Icon={ShieldAlert}
            label="Above threshold"
            value={`${overCount}`}
            unit={`/ ${rows.length} hosts`}
            tone={overCount > 0 ? "alert" : "good"}
            hint={`Risk at or above ${geometry.risk_threshold} this window`}
          />
          <Kpi
            Icon={Radar}
            label="Forecast horizon"
            value={`${(geometry.horizon_K * geometry.window_delta) / 60}`}
            unit="min"
            hint={`K=${geometry.horizon_K} windows of ${geometry.window_delta}s`}
          />
          <Kpi
            Icon={Cpu}
            label="State per host"
            value={`${geometry.n_features}`}
            unit="features"
            hint={`Context L=${geometry.context_L} · ${model.impl} pipeline`}
          />
        </div>

        {crossing && (
          <AlertStrip
            host={crossing.host}
            leadSeconds={crossing.forecast.lead_time_s as number}
            stage={crossing.forecast.observed_stage}
            risk={crossing.forecast.observed_risk}
          />
        )}

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[20rem_1fr]">
          <div className="flex flex-col gap-4">
            <Panel title="Host watch" bodyClass="">
              <HostWatch
                rows={rows}
                threshold={geometry.risk_threshold}
                selected={host}
                onSelect={setHost}
              />
            </Panel>
            <Panel title="Predicted stage mix">
              <StageMix
                horizons={current.horizons}
                selectedK={selectedK}
                onSelectK={setSelectedK}
                windowDeltaS={geometry.window_delta}
              />
            </Panel>
          </div>

          <div className="flex min-w-0 flex-col gap-4">
            <Panel
              title={`Forecast · ${current.host_id}`}
              aside={
                <label className="border-console-line text-console-muted hover:text-console-text flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors">
                  <input
                    type="checkbox"
                    checked={showReality}
                    onChange={(e) => setShowReality(e.target.checked)}
                    className="accent-observed-lit"
                  />
                  Show what actually happened
                </label>
              }
            >
              <div className="mb-3 flex flex-wrap items-center gap-x-6 gap-y-1.5 text-xs">
                {(
                  [
                    ["stage", current.observed_stage, false],
                    ["risk", current.observed_risk.toFixed(3), false],
                    ["origin", `${clockOf(current.origin_ts)} UTC`, false],
                    [
                      "lead time",
                      current.lead_time_s === null
                        ? "—"
                        : `${current.lead_time_s} s`,
                      current.lead_time_s !== null,
                    ],
                  ] as const
                ).map(([k, v, hot]) => (
                  <span key={k} className="text-console-muted">
                    {k}{" "}
                    <span
                      className={`font-mono tabular-nums ${
                        hot ? "text-threshold-lit font-semibold" : "text-console-text"
                      }`}
                    >
                      {v}
                    </span>
                  </span>
                ))}
              </div>
              <ForecastChart
                forecast={current}
                geometry={geometry}
                observed={observed}
                reality={reality}
                showReality={showReality}
              />
              {showReality && reality && (
                <p className="text-console-muted mt-2 text-[11px] leading-relaxed">
                  <span
                    className={
                      covered === reality.length
                        ? "text-positive-lit font-medium"
                        : "text-negative-lit font-medium"
                    }
                  >
                    {covered} of {reality.length} marks inside the band.
                  </span>{" "}
                  Crosses show the risk actually recorded at each horizon
                  timestamp, drawn where they fell. The band is a calibration
                  measurement, not a guarantee.
                </p>
              )}
            </Panel>

            <ReplayBar
              windowCount={windowCount}
              windowDeltaS={geometry.window_delta}
              t={ti}
              playing={playing}
              speed={speed}
              speeds={SPEEDS}
              label={`${clockOf(current.origin_ts)} UTC`}
              onSeek={(next) => {
                setPlayingOverride(false);
                setT(next);
              }}
              onPlayingChange={setPlayingOverride}
              onSpeedChange={setSpeed}
            />

            <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
              <Panel title="Attack lifecycle">
                <LifecycleTrack
                  observedStage={current.observed_stage}
                  predicted={kPoint}
                />
              </Panel>
              <Panel title="What is driving it">
                <Drivers signals={current.top_signals} explain={explanation} />
              </Panel>
            </div>
          </div>
        </div>

        <p className="text-console-muted pb-2 text-center text-[11px]">
          {sourceSummary}
        </p>
      </main>
    </div>
  );
}
