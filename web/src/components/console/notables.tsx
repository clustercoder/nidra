"use client";

import * as React from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, PanelRightOpen, Search } from "lucide-react";

import { clockOf, type Forecast } from "@/lib/demo-replay";
import { messageOf } from "./insight";
import { SEVERITY_ORDER, SeverityBadge, severityOf, severityRank } from "./severity";
import { FilterChip, ScopePill } from "./ui";

type Event = { host: string; forecast: Forecast; index: number; previousRisk?: number };
type SortKey = "time" | "host" | "stage" | "risk" | "lead";

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
      className={`border-console-line bg-console-surface sticky top-0 z-10 select-none border-b px-3 py-2 text-[10px] font-medium tracking-[0.08em] uppercase ${
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
 * Search / notable-event review — every window, every host, in one grid, with
 * the filters that narrowed it shown as chips so the reader always knows what
 * has been excluded from the count.
 */
export function Notables({
  byHost,
  hosts,
  threshold,
  query,
  onQueryChange,
  onlyAlerts,
  onOnlyAlertsChange,
  onOpen,
}: {
  byHost: Record<string, Forecast[]>;
  hosts: string[];
  threshold: number;
  query: string;
  onQueryChange: (q: string) => void;
  onlyAlerts: boolean;
  onOnlyAlertsChange: (v: boolean) => void;
  onOpen: (host: string, index: number) => void;
}) {
  const [sortKey, setSortKey] = React.useState<SortKey>("time");
  const [dir, setDir] = React.useState<"asc" | "desc">("desc");
  const [severity, setSeverity] = React.useState("all");

  const onSort = (k: SortKey) => {
    if (k === sortKey) setDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setDir("desc");
    }
  };

  const events = React.useMemo<Event[]>(() => {
    const out: Event[] = [];
    for (const host of hosts) {
      const list = byHost[host] ?? [];
      list.forEach((forecast, index) =>
        out.push({
          host,
          forecast,
          index,
          previousRisk: index > 0 ? list[index - 1].observed_risk : undefined,
        }),
      );
    }
    return out;
  }, [byHost, hosts]);

  const q = query.trim().toLowerCase();
  const filtered = events.filter((e) => {
    if (onlyAlerts && e.forecast.observed_risk < threshold) return false;
    if (severity !== "all" && severityOf(e.forecast.observed_stage) !== severity) return false;
    if (!q) return true;
    return (
      e.host.toLowerCase().includes(q) || e.forecast.observed_stage.toLowerCase().includes(q)
    );
  });

  const sorted = React.useMemo(() => {
    const mul = dir === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      switch (sortKey) {
        case "host":
          return mul * a.host.localeCompare(b.host);
        case "stage":
          return (
            mul * (severityRank(a.forecast.observed_stage) - severityRank(b.forecast.observed_stage))
          );
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

  const hasFilters = Boolean(q) || onlyAlerts || severity !== "all";

  return (
    <div className="border-console-line bg-console-surface flex min-h-0 flex-col overflow-hidden rounded-2xl border">
      <header className="flex flex-wrap items-center gap-3 px-4 pt-3.5 pb-3">
        <div className="min-w-0">
          <h2 className="text-console-text text-[13px] font-medium">Search results</h2>
          <p className="text-console-muted mt-0.5 text-[11px] tabular-nums">
            {sorted.length.toLocaleString()} of {events.length.toLocaleString()} windows across{" "}
            {hosts.length} hosts
          </p>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <ScopePill
            label="Severity"
            value={severity}
            onChange={setSeverity}
            options={[
              { value: "all", label: "All" },
              ...SEVERITY_ORDER.map((s) => ({ value: s, label: s })),
            ]}
          />
          <ScopePill
            label="Risk"
            value={onlyAlerts ? "over" : "any"}
            onChange={(v) => onOnlyAlertsChange(v === "over")}
            options={[
              { value: "any", label: "Any" },
              { value: "over", label: `≥ ${threshold}` },
            ]}
          />
        </div>
      </header>

      {hasFilters && (
        <div className="flex flex-wrap items-center gap-2 px-4 pb-3">
          {q && <FilterChip label="Query" value={query} onClear={() => onQueryChange("")} />}
          {severity !== "all" && (
            <FilterChip label="Severity" value={severity} onClear={() => setSeverity("all")} />
          )}
          {onlyAlerts && (
            <FilterChip
              label="Risk"
              value={`≥ ${threshold}`}
              onClear={() => onOnlyAlertsChange(false)}
            />
          )}
        </div>
      )}

      <div className="max-h-[calc(100vh-15rem)] min-h-0 overflow-x-auto overflow-y-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <Th label="Time" sortKey="time" active={sortKey === "time"} dir={dir} onSort={onSort} className="w-24" />
              <Th label="Host" sortKey="host" active={sortKey === "host"} dir={dir} onSort={onSort} />
              <Th label="Severity" sortKey="stage" active={sortKey === "stage"} dir={dir} onSort={onSort} />
              <th className="border-console-line bg-console-surface text-console-muted sticky top-0 z-10 border-b px-3 py-2 text-left text-[10px] font-medium tracking-[0.08em] uppercase">
                What happened
              </th>
              <Th label="Risk" sortKey="risk" active={sortKey === "risk"} dir={dir} onSort={onSort} align="right" />
              <Th label="Lead" sortKey="lead" active={sortKey === "lead"} dir={dir} onSort={onSort} align="right" />
              <th className="border-console-line bg-console-surface sticky top-0 z-10 border-b px-2 py-2" />
            </tr>
          </thead>
          <tbody className="divide-console-line/60 divide-y">
            {sorted.map((e) => {
              const over = e.forecast.observed_risk >= threshold;
              return (
                <tr
                  key={`${e.host}@${e.forecast.origin_ts}`}
                  onClick={() => onOpen(e.host, e.index)}
                  className="hover:bg-console-raised/40 group cursor-pointer transition-colors"
                >
                  <td className="text-console-muted px-3 py-2 font-mono text-[11px] whitespace-nowrap tabular-nums">
                    {clockOf(e.forecast.origin_ts)}
                  </td>
                  <td className="text-console-text px-3 py-2 font-mono text-[13px] whitespace-nowrap">
                    {e.host}
                  </td>
                  <td className="px-3 py-2">
                    <SeverityBadge stage={e.forecast.observed_stage} />
                  </td>
                  <td className="text-console-muted px-3 py-2">
                    {messageOf(e.forecast, threshold, e.previousRisk)}
                  </td>
                  <td
                    className={`px-3 py-2 text-right font-mono tabular-nums ${
                      over ? "text-threshold-lit font-semibold" : "text-console-text"
                    }`}
                  >
                    {e.forecast.observed_risk.toFixed(3)}
                  </td>
                  <td className="text-console-muted px-3 py-2 text-right font-mono whitespace-nowrap tabular-nums">
                    {e.forecast.lead_time_s !== null ? (
                      <span className="text-threshold-lit font-semibold">
                        {e.forecast.lead_time_s}s
                      </span>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-2 py-2 text-right">
                    <PanelRightOpen className="text-console-muted ml-auto size-3.5 opacity-0 transition-opacity group-hover:opacity-100" />
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={7} className="px-3 py-16 text-center">
                  <Search className="text-console-muted/50 mx-auto size-6" />
                  <p className="text-console-text mt-3 text-sm">No events match these filters</p>
                  <p className="text-console-muted mt-1 text-xs">
                    Clear a filter above to widen the search.
                  </p>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
