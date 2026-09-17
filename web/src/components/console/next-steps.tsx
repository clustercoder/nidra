"use client";

import { ArrowRight, CheckCircle2 } from "lucide-react";

import type { Recommendation } from "./insight";

/** "What to do next" — each card's button is wired to a real console action
 * (select the host, pivot to Search, toggle the outcome overlay). A button
 * that only looks like it works would be worse than no button on a demo. */
export function NextSteps({
  items,
  onAct,
}: {
  items: Recommendation[];
  onAct: (id: string) => void;
}) {
  return (
    <ul className="flex flex-col gap-2.5 p-4">
      {items.map((item) => {
        const done = item.tone === "done";
        return (
          <li
            key={item.id}
            className={`rounded-xl border p-3 ${
              done
                ? "border-positive-lit/25 bg-positive-lit/[0.06]"
                : "border-console-line bg-console-raised/40"
            }`}
          >
            <div className="flex items-start gap-2">
              {done ? (
                <CheckCircle2 className="text-positive-lit mt-0.5 size-3.5 shrink-0" />
              ) : (
                <span className="bg-threshold-lit mt-1.5 size-1.5 shrink-0 rounded-full" />
              )}
              <div className="min-w-0 flex-1">
                <p className="text-console-text text-[13px] font-medium">{item.title}</p>
                <p className="text-console-muted mt-1 text-xs leading-relaxed">{item.body}</p>
                <button
                  onClick={() => onAct(item.id)}
                  className="border-console-line bg-console-surface text-console-text hover:border-observed-lit/50 hover:text-observed-lit mt-2.5 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors"
                >
                  {item.actionLabel}
                  <ArrowRight className="size-3" />
                </button>
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
