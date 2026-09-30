"use client";

import * as React from "react";
import { Activity, ListChecks, Radar, Server, ShieldAlert, Timer, Waypoints } from "lucide-react";

import { ActivityFeed, type FeedItem } from "@/components/console/activity-feed";
import { CaptureUpload } from "@/components/console/capture-upload";
import { DetailDrawer } from "@/components/console/detail-drawer";
import { ChartLegend, ForecastChart } from "@/components/console/forecast-chart";
import { PostureHero } from "@/components/console/hero";
import { HostTable } from "@/components/console/host-table";
import {
  isNotable,
  messageOf,
  postureOf,
  recommendationsOf,
  trendOf,
} from "@/components/console/insight";
import { NextSteps } from "@/components/console/next-steps";
import { Notables } from "@/components/console/notables";
import {
  ConsoleRail,
  ConsoleTopBar,
  PageHeader,
  type ConsoleView,
} from "@/components/console/shell";
import { StatTile } from "@/components/console/ui";
import {
  AlertStrip,
  Drivers,
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
  type UploadedReplay,
} from "@/lib/demo-replay";

const SPEEDS = [1, 10, 30, 60] as const;
const DEFAULT_SPEED = 30;
const RING_CAP = 200;
const SPARK_WINDOWS = 24;
/* Deep enough that a change-filtered feed still has something in it during a
   quiet stretch, shallow enough to stay "recent". */
const FEED_WINDOWS = 12;

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
  uploaded,
  onAnalysed,
  onClearUpload,
}: {
  model: ModelInfo;
  hosts: string[];
  byHost: Record<string, Forecast[]>;
  explanations: Record<string, ExplainInfo>;
  sourceSummary: string;
  initial: { host?: string; t: number; paused: boolean };
  /* Present when the console is showing an analysed upload rather than the
     committed replay. The console renders both identically; this is only what
     the capture view needs to describe what is on screen. */
  uploaded?: UploadedReplay | null;
  onAnalysed?: (replay: UploadedReplay) => void;
  onClearUpload?: () => void;
}) {
  const geometry = model.config;
  const windowCount = byHost[hosts[0]]?.length ?? 0;
  const lastIndex = Math.max(0, windowCount - 1);

  const escalating = React.useMemo(
    () => hosts.find((h) => (byHost[h] ?? []).some((f) => f.lead_time_s !== null)) ?? hosts[0],
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
  const [view, setView] = React.useState<ConsoleView>("dashboard");
  const [query, setQuery] = React.useState("");
  const [onlyAlerts, setOnlyAlerts] = React.useState(false);
  const [detail, setDetail] = React.useState<{ host: string; index: number } | null>(null);

  /* One rAF loop drives the clock: wall time scaled by the replay speed and
     converted to whole windows, so the default 30x advances one 30s window a
     second rather than re-rendering on every frame. 60x is selectable but not
     the default — at two windows a second the risk figures move faster than
     anyone can read them. It holds while a record is open — reading a detail
     whose numbers move underneath you is unusable. */
  React.useEffect(() => {
    if (!playing || windowCount === 0 || detail) return;
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
  }, [playing, speed, windowCount, geometry.window_delta, detail]);

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
        The replay fixture is empty — regenerate it with scripts/make_demo_replay.py.
      </p>
    );
  }

  const ti = Math.min(t, lastIndex);
  const rows = hosts
    .map((h) => ({
      host: h,
      forecast: byHost[h]?.[ti],
      series: (riskSeries[h] ?? []).slice(Math.max(0, ti + 1 - SPARK_WINDOWS), ti + 1),
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
  const overCount = rows.filter((r) => r.forecast.observed_risk >= geometry.risk_threshold).length;
  const crossing = rows
    .filter((r) => r.forecast.lead_time_s !== null)
    .sort((a, b) => (a.forecast.lead_time_s ?? 0) - (b.forecast.lead_time_s ?? 0))[0];
  const kPoint = current.horizons.find((h) => h.k === selectedK) ?? current.horizons[0];

  const posture = postureOf({
    rows,
    geometry,
    crossingHost: crossing?.host,
    leadSeconds: crossing?.forecast.lead_time_s,
  });
  const recommendations = recommendationsOf({
    rows,
    geometry,
    crossingHost: crossing?.host,
    leadSeconds: crossing?.forecast.lead_time_s,
    showingReality: showReality,
  });

  const q = query.trim().toLowerCase();
  const tableRows = q
    ? rows.filter(
        (r) =>
          r.host.toLowerCase().includes(q) || r.forecast.observed_stage.toLowerCase().includes(q),
      )
    : rows;

  /* The feed reads across hosts, unlike everything else on the dashboard, and
     carries only windows that changed something — see isNotable. */
  const feedItems: FeedItem[] = [];
  for (let step = 0; step < FEED_WINDOWS; step += 1) {
    const idx = ti - step;
    if (idx < 0) break;
    for (const h of hosts) {
      const f = byHost[h]?.[idx];
      if (!f) continue;
      const prev = idx > 0 ? byHost[h]?.[idx - 1] : undefined;
      if (!isNotable(f, prev, geometry.risk_threshold)) continue;
      feedItems.push({ host: h, forecast: f, index: idx, previousRisk: prev?.observed_risk });
    }
  }
  feedItems.sort((a, b) => {
    const dt = Date.parse(b.forecast.origin_ts) - Date.parse(a.forecast.origin_ts);
    return dt !== 0 ? dt : b.forecast.observed_risk - a.forecast.observed_risk;
  });

  const currentPrevRisk = ti > 0 ? hostList[ti - 1]?.observed_risk : undefined;
  const currentTrend = trendOf(current.observed_risk, currentPrevRisk, geometry.risk_threshold);

  const openDetail = (h: string, index: number) => {
    setPlayingOverride(false);
    setDetail({ host: h, index });
  };

  const onRecommendation = (id: string) => {
    if (id === "focus-crossing" && crossing) {
      setHost(crossing.host);
      setPlayingOverride(false);
      setView("dashboard");
    } else if (id === "review-over" || id === "fleet-clear") {
      setOnlyAlerts(id === "review-over");
      setView("search");
    } else if (id === "validate") {
      setShowReality((v) => !v);
    }
  };

  const detailForecast = detail ? byHost[detail.host]?.[detail.index] : undefined;

  return (
    <div className="bg-console-bg text-console-text flex h-screen flex-col overflow-hidden">
      <div className="flex min-h-0 flex-1">
        <ConsoleRail view={view} onViewChange={setView} alertCount={overCount} />

        <div className="flex min-w-0 flex-1 flex-col">
          <ConsoleTopBar
            breadcrumb={view === "dashboard" ? "Dashboard" : "Search"}
            query={query}
            onQueryChange={setQuery}
            modelLabel={`${model.impl} · ${model.model_version}`}
          />

          <main className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
            {view === "capture" ? (
              <>
                <PageHeader
                  title="Analyse a capture"
                  subtitle="Score a pcap the model has never seen, through the same ensemble this console is already running."
                />
                <CaptureUpload
                  active={uploaded ?? null}
                  onAnalysed={(r) => {
                    onAnalysed?.(r);
                    setView("dashboard");
                  }}
                  onClear={() => {
                    onClearUpload?.();
                    setView("dashboard");
                  }}
                />
              </>
            ) : view === "search" ? (
              <>
                <PageHeader
                  title="Search"
                  subtitle={`Every forecast window in the replay — ${windowCount} per host across ${hosts.length} hosts.`}
                />
                <Notables
                  byHost={byHost}
                  hosts={hosts}
                  threshold={geometry.risk_threshold}
                  query={query}
                  onQueryChange={setQuery}
                  onlyAlerts={onlyAlerts}
                  onOnlyAlertsChange={setOnlyAlerts}
                  onOpen={openDetail}
                />
              </>
            ) : (
              <>
                <PageHeader
                  title="Forecast console"
                  subtitle="Replay of real CIC-IDS2017 traffic scored by the trained ensemble — risk, projected stage, and lead time per host."
                >
                  <span className="border-console-line bg-console-surface text-console-muted inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs">
                    <span
                      className={`size-1.5 rounded-full ${
                        playing ? "bg-positive-lit animate-pulse" : "bg-console-muted"
                      }`}
                    />
                    {playing ? "Replaying" : "Paused"}
                    <span className="text-console-text font-mono tabular-nums">
                      {clockOf(current.origin_ts)}
                    </span>
                  </span>
                </PageHeader>

                {crossing && (
                  <AlertStrip
                    host={crossing.host}
                    leadSeconds={crossing.forecast.lead_time_s as number}
                    stage={crossing.forecast.observed_stage}
                    risk={crossing.forecast.observed_risk}
                  />
                )}

                {/* posture, the chart it is claimed from, and what to do about it */}
                <div className="grid gap-4 xl:grid-cols-12">
                  <div className="xl:col-span-3">
                    <PostureHero
                      posture={posture}
                      originTs={current.origin_ts}
                      hostCount={rows.length}
                      aboveCount={overCount}
                      threshold={geometry.risk_threshold}
                      primaryLabel={`Open ${crossing?.host ?? host}`}
                      onPrimary={() => openDetail(crossing?.host ?? host, ti)}
                    />
                  </div>

                  <div className="xl:col-span-6">
                    <Panel
                      title={`Forecast · ${current.host_id}`}
                      subtitle={`${current.observed_stage} · risk ${current.observed_risk.toFixed(3)} · ${
                        current.lead_time_s === null
                          ? "no crossing inside horizon"
                          : `crossing in ${current.lead_time_s}s`
                      }`}
                      Icon={Waypoints}
                      iconTone={current.lead_time_s === null ? "observed" : "alert"}
                      className="h-full"
                      aside={
                        <label className="border-console-line hover:border-observed-lit/50 text-console-muted hover:text-console-text flex cursor-pointer items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-medium transition-colors">
                          <input
                            type="checkbox"
                            checked={showReality}
                            onChange={(e) => setShowReality(e.target.checked)}
                            className="accent-observed-lit"
                          />
                          Show outcome
                        </label>
                      }
                    >
                      <ForecastChart
                        forecast={current}
                        geometry={geometry}
                        observed={observed}
                        reality={reality}
                        showReality={showReality}
                      />
                      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                        <ChartLegend showReality={showReality} />
                        {showReality && reality && (
                          <p className="text-console-muted text-[11px]">
                            <span
                              className={
                                covered === reality.length
                                  ? "text-positive-lit font-medium"
                                  : "text-negative-lit font-medium"
                              }
                            >
                              {covered} of {reality.length} inside the band
                            </span>{" "}
                            — drawn where they fell.
                          </p>
                        )}
                      </div>
                    </Panel>
                  </div>

                  <div className="xl:col-span-3">
                    <Panel
                      title="What to do next"
                      subtitle="Each action drives this console"
                      Icon={ListChecks}
                      iconTone="projected"
                      bodyClass=""
                      className="h-full"
                    >
                      <NextSteps items={recommendations} onAct={onRecommendation} />
                    </Panel>
                  </div>
                </div>

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

                <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                  <StatTile
                    Icon={Timer}
                    label="Earliest warning"
                    value={crossing ? String(crossing.forecast.lead_time_s) : "None"}
                    unit={crossing ? "seconds" : undefined}
                    tone={crossing ? "alert" : "good"}
                    noteTone={crossing ? "alert" : "good"}
                    note={
                      crossing
                        ? `${crossing.host} crosses ${geometry.risk_threshold} first`
                        : "No crossing projected this window"
                    }
                  />
                  <StatTile
                    Icon={ShieldAlert}
                    label="Above threshold"
                    value={String(overCount)}
                    unit={`of ${rows.length} hosts`}
                    tone={overCount > 0 ? "alert" : "good"}
                    noteTone={overCount > 0 ? "alert" : "good"}
                    note={`Risk at or above ${geometry.risk_threshold}`}
                  />
                  <StatTile
                    Icon={Activity}
                    label={`Risk · ${current.host_id}`}
                    value={current.observed_risk.toFixed(3)}
                    tone={current.observed_risk >= geometry.risk_threshold ? "alert" : "observed"}
                    noteTone={
                      currentTrend === "rising"
                        ? "bad"
                        : currentTrend === "falling"
                          ? "good"
                          : "neutral"
                    }
                    note={`${currentTrend} across recent windows`}
                    series={riskSeries[host]?.slice(Math.max(0, ti + 1 - SPARK_WINDOWS), ti + 1)}
                  />
                  <StatTile
                    Icon={Radar}
                    label="Forecast horizon"
                    value={String((geometry.horizon_K * geometry.window_delta) / 60)}
                    unit="minutes"
                    tone="projected"
                    note={`K=${geometry.horizon_K} windows of ${geometry.window_delta}s`}
                  />
                </div>

                <div className="grid gap-4 xl:grid-cols-12">
                  <div className="xl:col-span-5">
                    <Panel
                      title="Host watch"
                      subtitle={
                        query
                          ? `${tableRows.length} of ${rows.length} match “${query}”`
                          : "This window, every host"
                      }
                      Icon={Server}
                      iconTone="observed"
                      bodyClass="pb-2"
                      className="h-full"
                    >
                      <HostTable
                        rows={tableRows}
                        threshold={geometry.risk_threshold}
                        selected={host}
                        onSelect={setHost}
                        onOpen={(h) => openDetail(h, ti)}
                      />
                    </Panel>
                  </div>

                  <div className="xl:col-span-7">
                    <Panel
                      title="Fleet activity"
                      subtitle={`Changes over the last ${FEED_WINDOWS} windows, all hosts`}
                      Icon={Activity}
                      iconTone="projected"
                      bodyClass="pb-2"
                      className="h-full"
                    >
                      <ActivityFeed
                        items={feedItems.slice(0, 8)}
                        threshold={geometry.risk_threshold}
                        onSelect={(item) => openDetail(item.host, item.index)}
                      />
                    </Panel>
                  </div>
                </div>

                <div className="grid gap-4 xl:grid-cols-12">
                  {/* the two short panels stack so they do not leave a well
                      beside the taller driver list */}
                  <div className="flex flex-col gap-4 xl:col-span-5">
                    <Panel
                      title="Attack lifecycle"
                      subtitle={messageOf(current, geometry.risk_threshold, currentPrevRisk)}
                      Icon={Waypoints}
                      iconTone="observed"
                    >
                      <LifecycleTrack observedStage={current.observed_stage} predicted={kPoint} />
                    </Panel>
                    <Panel
                      title="Predicted stage mix"
                      subtitle={`${current.host_id} at +${kPoint.k * geometry.window_delta}s`}
                      Icon={Radar}
                      iconTone="projected"
                      className="flex-1"
                    >
                      <StageMix
                        horizons={current.horizons}
                        selectedK={selectedK}
                        onSelectK={setSelectedK}
                        windowDeltaS={geometry.window_delta}
                      />
                    </Panel>
                  </div>
                  <div className="xl:col-span-7">
                    <Panel
                      title="What is driving it"
                      subtitle="Signed contribution per signal at this window"
                      Icon={Activity}
                      iconTone="good"
                      className="h-full"
                    >
                      <Drivers signals={current.top_signals} explain={explanation} />
                    </Panel>
                  </div>
                </div>

                <p className="text-console-muted pb-2 text-center text-[11px] leading-relaxed">
                  {sourceSummary}
                </p>
              </>
            )}
          </main>
        </div>
      </div>

      {detail && detailForecast && (
        <DetailDrawer
          host={detail.host}
          forecast={detailForecast}
          model={model}
          explanation={explanations[`${detail.host}@${detailForecast.origin_ts}`]}
          threshold={geometry.risk_threshold}
          previousRisk={
            detail.index > 0 ? byHost[detail.host]?.[detail.index - 1]?.observed_risk : undefined
          }
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  );
}
