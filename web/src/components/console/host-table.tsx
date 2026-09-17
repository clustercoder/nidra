"use client";

import * as React from "react";
import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";

import type { Forecast } from "@/lib/demo-replay";
import { SeverityBadge, severityRank } from "./severity";

type Row = { host: string; forecast: Forecast; series: number[] };
type SortKey = "host" | "risk" | "stage" | "lead";

function Spark({ series, over }: { series: number[]; over: boolean }) {
  if (series.length < 2) return <span className="block h-5 w-16" />;
  const w = 64;
  const h = 20;
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
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Th({
  label,
  sortKey,
  active,
  dir,
  onSort,
  align = "left",
}: {
  label: string;
  sortKey: SortKey;
  active: boolean;
  dir: "asc" | "desc";
  onSort: (k: SortKey) => void;
  align?: "left" | "right";
}) {
  return (
    <th
      className={`border-console-line bg-console-raised/60 select-none border-b px-3 py-1.5 text-[10px] font-semibold tracking-[0.1em] uppercase ${
        align === "right" ? "text-right" : "text-left"
      }`}
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

/** Dense, sortable event grid — the search-results-table treatment, standing
 * in for the old card list so hosts read as a scannable table, not a stack of
 * tiles. */
export function HostTable({
  rows,
  threshold,
  selected,
  onSelect,
}: {
  rows: Row[];
  threshold: number;
  selected: string;
  onSelect: (host: string) => void;
}) {
  const [sortKey, setSortKey] = React.useState<SortKey>("risk");
  const [dir, setDir] = React.useState<"asc" | "desc">("desc");

  const onSort = (k: SortKey) => {
    if (k === sortKey) {
      setDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(k);
      setDir("desc");
    }
  };

  const sorted = React.useMemo(() => {
    const mul = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      switch (sortKey) {
        case "host":
          return mul * a.host.localeCompare(b.host);
        case "stage":
          return mul * (severityRank(a.forecast.observed_stage) - severityRank(b.forecast.observed_stage));
        case "lead": {
          const av = a.forecast.lead_time_s ?? Infinity;
          const bv = b.forecast.lead_time_s ?? Infinity;
          return mul * (av - bv);
        }
        case "risk":
        default:
          return mul * (a.forecast.observed_risk - b.forecast.observed_risk);
      }
    });
  }, [rows, sortKey, dir]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <Th label="Host" sortKey="host" active={sortKey === "host"} dir={dir} onSort={onSort} />
            <Th label="Stage" sortKey="stage" active={sortKey === "stage"} dir={dir} onSort={onSort} />
            <th className="border-console-line bg-console-raised/60 border-b px-3 py-1.5 text-left text-[10px] font-semibold tracking-[0.1em] uppercase text-console-muted">
              Trend
            </th>
            <Th label="Risk" sortKey="risk" active={sortKey === "risk"} dir={dir} onSort={onSort} align="right" />
            <Th label="Lead" sortKey="lead" active={sortKey === "lead"} dir={dir} onSort={onSort} align="right" />
          </tr>
        </thead>
        <tbody className="divide-console-line divide-y">
          {sorted.map(({ host, forecast, series }) => {
            const over = forecast.observed_risk >= threshold;
            const active = host === selected;
            return (
              <tr
                key={host}
                onClick={() => onSelect(host)}
                aria-current={active ? "true" : undefined}
                className={`cursor-pointer transition-colors ${
                  active
                    ? "bg-console-raised border-l-2 border-l-observed-lit"
                    : "hover:bg-console-raised/50 border-l-2 border-l-transparent"
                }`}
              >
                <td className="px-3 py-2 font-mono text-[13px] text-console-text whitespace-nowrap">
                  {host}
                </td>
                <td className="px-3 py-2">
                  <SeverityBadge stage={forecast.observed_stage} />
                </td>
                <td className="px-3 py-2">
                  <Spark series={series} over={over} />
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono tabular-nums ${
                    over ? "text-threshold-lit font-semibold" : "text-console-muted"
                  }`}
                >
                  {forecast.observed_risk.toFixed(3)}
                </td>
                <td className="px-3 py-2 text-right font-mono tabular-nums text-console-muted whitespace-nowrap">
                  {forecast.lead_time_s !== null ? (
                    <span className="text-threshold-lit font-semibold">
                      {forecast.lead_time_s}s
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
