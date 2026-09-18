"use client";

import { ArrowDownRight, ArrowUpRight, Minus, ShieldAlert, Timer } from "lucide-react";
import type { LucideIcon } from "lucide-react";

import { clockOf, type Forecast } from "@/lib/demo-replay";
import { messageOf, trendOf } from "./insight";
import { severityOf } from "./severity";
import { IconChip, type Tone } from "./ui";

export type FeedItem = {
  host: string;
  forecast: Forecast;
  index: number;
  previousRisk?: number;
};

function glyphOf(item: FeedItem, threshold: number): { Icon: LucideIcon; tone: Tone } {
  if (item.forecast.lead_time_s !== null) return { Icon: Timer, tone: "alert" };
  if (item.forecast.observed_risk >= threshold) return { Icon: ShieldAlert, tone: "alert" };
  const trend = trendOf(item.forecast.observed_risk, item.previousRisk, threshold);
  if (trend === "rising") return { Icon: ArrowUpRight, tone: "projected" };
  if (trend === "falling") return { Icon: ArrowDownRight, tone: "good" };
  return { Icon: Minus, tone: "neutral" };
}

/** Recent windows across the whole fleet, newest first — the running account
 * of the replay that the per-host chart cannot give on its own. */
export function ActivityFeed({
  items,
  threshold,
  onSelect,
}: {
  items: FeedItem[];
  threshold: number;
  onSelect: (item: FeedItem) => void;
}) {
  if (items.length === 0) {
    return (
      <div className="px-4 py-10 text-center">
        <Minus className="text-console-muted/50 mx-auto size-5" />
        <p className="text-console-text mt-2 text-xs">No change across the fleet</p>
        <p className="text-console-muted mt-1 text-[11px]">
          Every host held its stage and risk over these windows.
        </p>
      </div>
    );
  }

  return (
    <ul className="divide-console-line/70 divide-y">
      {items.map((item) => {
        const { Icon, tone } = glyphOf(item, threshold);
        const sev = severityOf(item.forecast.observed_stage);
        return (
          <li key={`${item.host}@${item.forecast.origin_ts}`}>
            <button
              onClick={() => onSelect(item)}
              className="hover:bg-console-raised/50 flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors"
            >
              <IconChip Icon={Icon} tone={tone} size="sm" />
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline gap-2">
                  <span className="text-console-text truncate font-mono text-[13px]">
                    {item.host}
                  </span>
                  {sev !== "info" && (
                    <span className="text-console-muted text-[10px] tracking-wide uppercase">
                      {sev}
                    </span>
                  )}
                </span>
                <span className="text-console-muted mt-0.5 block truncate text-xs">
                  {messageOf(item.forecast, threshold, item.previousRisk)}
                </span>
              </span>
              <span className="text-console-muted shrink-0 font-mono text-[11px] tabular-nums">
                {clockOf(item.forecast.origin_ts)}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}
