"use client";

import * as React from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, PanelRightOpen } from "lucide-react";

import type { Forecast } from "@/lib/demo-replay";
import { SeverityBadge, severityRank } from "./severity";
import { Sparkline } from "./ui";

type Row = { host: string; forecast: Forecast; series: number[] };
type SortKey = "host" | "risk" | "stage" | "lead";

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
      className={`border-console-line select-none border-b px-3 pb-2 text-[10px] font-medium tracking-[0.08em] uppercase ${
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

/** Fleet grid. Row click selects the host the chart draws; the trailing button
 * opens the full record, the way an incident queue splits "look at this" from
 * "open this". */
export function HostTable({
  rows,
  threshold,
  selected,
  onSelect,
  onOpen,
}: {
  rows: Row[];
  threshold: number;
  selected: string;
  onSelect: (host: string) => void;
  onOpen: (host: string) => void;
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
          return (
            mul * (severityRank(a.forecast.observed_stage) - severityRank(b.forecast.observed_stage))
          );
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

  if (rows.length === 0) {
    return (
      <p className="text-console-muted px-4 py-8 text-center text-xs">No host matches this search.</p>
    );
  }

  return (
    <div className="overflow-x-auto px-1 pb-1">
      <table className="w-full border-collapse text-xs">
        <thead>
          <tr>
            <Th label="Host" sortKey="host" active={sortKey === "host"} dir={dir} onSort={onSort} />
            <Th label="Stage" sortKey="stage" active={sortKey === "stage"} dir={dir} onSort={onSort} />
            <th className="border-console-line text-console-muted border-b px-3 pb-2 text-left text-[10px] font-medium tracking-[0.08em] uppercase">
              Trend
            </th>
            <Th label="Risk" sortKey="risk" active={sortKey === "risk"} dir={dir} onSort={onSort} align="right" />
            <Th label="Lead" sortKey="lead" active={sortKey === "lead"} dir={dir} onSort={onSort} align="right" />
            <th className="border-console-line border-b px-2 pb-2" />
          </tr>
        </thead>
        <tbody>
          {sorted.map(({ host, forecast, series }) => {
            const over = forecast.observed_risk >= threshold;
            const active = host === selected;
            return (
              <tr
                key={host}
                onClick={() => onSelect(host)}
                aria-current={active ? "true" : undefined}
                className={`group cursor-pointer transition-colors ${
                  active ? "bg-console-raised/70" : "hover:bg-console-raised/40"
                }`}
              >
                <td className="rounded-l-lg py-2 pr-3 pl-3 whitespace-nowrap">
                  <span className="flex items-center gap-2">
                    <span
                      className={`h-6 w-0.5 shrink-0 rounded-full ${
                        active ? "bg-observed-lit" : "bg-transparent"
                      }`}
                    />
                    <span className="text-console-text font-mono text-[13px]">{host}</span>
                  </span>
                </td>
                <td className="px-3 py-2">
                  <SeverityBadge stage={forecast.observed_stage} />
                </td>
                <td className="px-3 py-2">
                  <Sparkline series={series} tone={over ? "alert" : "observed"} className="h-6 w-20" />
                </td>
                <td
                  className={`px-3 py-2 text-right font-mono tabular-nums ${
                    over ? "text-threshold-lit font-semibold" : "text-console-text"
                  }`}
                >
                  {forecast.observed_risk.toFixed(3)}
                </td>
                <td className="text-console-muted px-3 py-2 text-right font-mono whitespace-nowrap tabular-nums">
                  {forecast.lead_time_s !== null ? (
                    <span className="text-threshold-lit font-semibold">{forecast.lead_time_s}s</span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="rounded-r-lg px-2 py-2 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpen(host);
                    }}
                    aria-label={`Open detail for ${host}`}
                    className="text-console-muted hover:bg-console-surface hover:text-console-text rounded-md p-1 opacity-0 transition-all group-hover:opacity-100 focus-visible:opacity-100"
                  >
                    <PanelRightOpen className="size-3.5" />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
