"use client";

import * as React from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

import { clockOf, type Forecast } from "@/lib/demo-replay";
import { SeverityBadge, severityRank } from "./severity";

type Event = { host: string; forecast: Forecast; index: number };
type SortKey = "time" | "host" | "stage" | "risk" | "lead";

function messageOf(f: Forecast): string {
  if (f.lead_time_s !== null) {
    return `risk ${f.observed_risk.toFixed(3)} · projected to cross threshold in ${f.lead_time_s}s`;
  }
  return `risk ${f.observed_risk.toFixed(3)} · no threshold crossing inside horizon`;
}

function Th({
  label,
  sortKey,
  active,
  dir,
  onSort,
  align = "left",
  className = "",
}: {
  label: string;
  sortKey: SortKey;
  active: boolean;
  dir: "asc" | "desc";
  onSort: (k: SortKey) => void;
  align?: "left" | "right";
  className?: string;
}) {
  return (
    <th
      className={`border-console-line bg-console-raised/60 sticky top-0 select-none border-b px-3 py-1.5 text-[10px] font-semibold tracking-[0.1em] uppercase ${
        align === "right" ? "text-right" : "text-left"
      } ${className}`}
    >
      <button
        onClick={() => onSort(sortKey)}
        className={`inline-flex items-center gap-1 transition-colors ${
          active ? "text-console-text" : "text-console-muted hover:text-console-text"
        } ${align === "right" ? "flex-row-reverse" : ""}`}
      >
        {label}
        {active ? (
          dir === "asc" ? (
            <ArrowUp className="size-3" />
          ) : (
            <ArrowDown className="size-3" />
          )
        ) : (
          <ArrowUpDown className="size-3 opacity-40" />
        )}
      </button>
    </th>
  );
}

/**
 * Search / notable-event review — every window, every host, flattened into
 * one scrollable grid. This is what "Search" means in the rail: the same
 * fixture the dashboard shows one slice of, laid out the way Splunk ES lists
 * notables, so the whole replay is auditable rather than only its current
 * instant.
 */
export function Notables({
  byHost,
  hosts,
  threshold,
  query,
  onJump,
}: {
  byHost: Record<string, Forecast[]>;
  hosts: string[];
  threshold: number;
  query: string;
  onJump: (host: string, index: number) => void;
}) {
  const [sortKey, setSortKey] = React.useState<SortKey>("time");
  const [dir, setDir] = React.useState<"asc" | "desc">("desc");
  const [onlyAlerts, setOnlyAlerts] = React.useState(false);

  const onSort = (k: SortKey) => {
    if (k === sortKey) {
      setDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(k);
      setDir(k === "time" ? "desc" : "desc");
    }
  };

  const events = React.useMemo<Event[]>(() => {
    const out: Event[] = [];
    for (const host of hosts) {
      (byHost[host] ?? []).forEach((forecast, index) => out.push({ host, forecast, index }));
    }
    return out;
  }, [byHost, hosts]);

  const q = query.trim().toLowerCase();
  const filtered = events.filter((e) => {
    if (onlyAlerts && e.forecast.observed_risk < threshold) return false;
    if (!q) return true;
    return (
      e.host.toLowerCase().includes(q) ||
      e.forecast.observed_stage.toLowerCase().includes(q)
    );
  });

  const sorted = React.useMemo(() => {
    const mul = dir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      switch (sortKey) {
        case "host":
          return mul * a.host.localeCompare(b.host);
        case "stage":
          return mul * (severityRank(a.forecast.observed_stage) - severityRank(b.forecast.observed_stage));
        case "risk":
          return mul * (a.forecast.observed_risk - b.forecast.observed_risk);
        case "lead": {
          const av = a.forecast.lead_time_s ?? Infinity;
          const bv = b.forecast.lead_time_s ?? Infinity;
          return mul * (av - bv);
        }
        case "time":
        default:
          return mul * (Date.parse(a.forecast.origin_ts) - Date.parse(b.forecast.origin_ts));
      }
    });
  }, [filtered, sortKey, dir]);

  return (
    <div className="border-console-line bg-console-surface flex flex-col overflow-hidden rounded-md border">
      <header className="border-console-line flex flex-wrap items-center gap-3 border-b px-4 py-2.5">
        <h2 className="text-console-muted text-[11px] font-semibold tracking-[0.14em] uppercase">
          Search results
        </h2>
        <span className="text-console-muted font-mono text-[11px] tabular-nums">
          {sorted.length} of {events.length} events
        </span>
        <label className="border-console-line text-console-muted hover:text-console-text ml-auto flex cursor-pointer items-center gap-2 rounded border px-2 py-1 text-[11px] font-medium transition-colors">
          <input
            type="checkbox"
            checked={onlyAlerts}
            onChange={(e) => setOnlyAlerts(e.target.checked)}
            className="accent-threshold-lit"
          />
          At/above threshold only
        </label>
      </header>
      <div className="max-h-[560px] overflow-x-auto overflow-y-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <Th label="Time" sortKey="time" active={sortKey === "time"} dir={dir} onSort={onSort} className="w-24" />
              <Th label="Host" sortKey="host" active={sortKey === "host"} dir={dir} onSort={onSort} />
              <Th label="Severity" sortKey="stage" active={sortKey === "stage"} dir={dir} onSort={onSort} />
              <th className="border-console-line bg-console-raised/60 sticky top-0 border-b px-3 py-1.5 text-left text-[10px] font-semibold tracking-[0.1em] uppercase text-console-muted">
                Message
              </th>
              <Th label="Risk" sortKey="risk" active={sortKey === "risk"} dir={dir} onSort={onSort} align="right" />
              <Th label="Lead" sortKey="lead" active={sortKey === "lead"} dir={dir} onSort={onSort} align="right" />
            </tr>
          </thead>
          <tbody className="divide-console-line divide-y">
            {sorted.map((e) => {
              const over = e.forecast.observed_risk >= threshold;
              return (
                <tr
                  key={`${e.host}@${e.forecast.origin_ts}`}
                  onClick={() => onJump(e.host, e.index)}
                  className="hover:bg-console-raised/50 cursor-pointer transition-colors"
                >
                  <td className="px-3 py-1.5 font-mono text-[11px] whitespace-nowrap text-console-muted">
                    {clockOf(e.forecast.origin_ts)}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-[13px] whitespace-nowrap text-console-text">
                    {e.host}
                  </td>
                  <td className="px-3 py-1.5">
                    <SeverityBadge stage={e.forecast.observed_stage} />
                  </td>
                  <td className="px-3 py-1.5 text-console-muted">{messageOf(e.forecast)}</td>
                  <td
                    className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                      over ? "text-threshold-lit font-semibold" : "text-console-muted"
                    }`}
                  >
                    {e.forecast.observed_risk.toFixed(3)}
                  </td>
                  <td className="px-3 py-1.5 text-right font-mono tabular-nums whitespace-nowrap text-console-muted">
                    {e.forecast.lead_time_s !== null ? (
                      <span className="text-threshold-lit font-semibold">{e.forecast.lead_time_s}s</span>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={6} className="text-console-muted px-3 py-8 text-center">
                  No events match this search.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
