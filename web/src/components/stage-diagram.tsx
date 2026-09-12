import { ArrowRight, Sparkle } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type NodeColor = "blue" | "lavender" | "mint";

export interface DiagramNode {
  /** A lucide icon, or a short literal to set inside the circle ("30 s", "45-d"). */
  icon: LucideIcon | string;
  label: string;
  sublabel: string;
  color: NodeColor;
}

const RING: Record<NodeColor, string> = {
  blue: "bg-node-blue/40 border-observed text-observed",
  lavender: "bg-node-lavender/40 border-projected text-projected",
  mint: "bg-node-mint/40 border-positive text-positive",
};

/**
 * The three-node stage diagram: a circle per step, connected left to right.
 * Replaces the captured product screenshots — everything here is drawn, so it
 * stays honest and stays in the token palette.
 */
export function StageDiagram({
  title,
  nodes,
  className = "",
}: {
  title: string;
  nodes: DiagramNode[];
  className?: string;
}) {
  return (
    <div
      className={`border-gray-light flex h-full flex-col overflow-hidden rounded-xl border bg-white ${className}`}
    >
      <div className="border-gray-light flex items-center gap-2 border-b px-5 py-3">
        <Sparkle className="text-primary-blue size-4" strokeWidth={1.75} aria-hidden />
        <span className="text-gray-black-soft text-base font-semibold">{title}</span>
      </div>
      <div className="dot-grid flex flex-1 flex-col items-stretch justify-center gap-2 px-5 py-8 md:flex-row md:items-center md:justify-between">
        {nodes.map((node, i) => {
          const isText = typeof node.icon === "string";
          const Icon = isText ? null : (node.icon as LucideIcon);
          return (
            <div key={node.label} className="contents">
              <div className="flex flex-1 flex-col items-center text-center">
                <div
                  className={`flex size-28 shrink-0 items-center justify-center rounded-full border-2 ${RING[node.color]}`}
                >
                  {Icon ? (
                    <Icon className="size-9" strokeWidth={1.5} />
                  ) : (
                    <span className="font-headings text-xl font-semibold">
                      {node.icon as string}
                    </span>
                  )}
                </div>
                <div className="text-gray-black-soft mt-4 max-w-[190px] text-base font-semibold">
                  {node.label}
                </div>
                <div className="text-gray-dark mt-1.5 max-w-[190px] text-sm leading-snug">
                  {node.sublabel}
                </div>
              </div>
              {i < nodes.length - 1 && (
                <div
                  className="text-gray-medium flex shrink-0 items-center justify-center py-3 md:-mt-[52px] md:py-0"
                  aria-hidden
                >
                  <span className="bg-gray-medium hidden h-px w-6 md:block" />
                  <ArrowRight className="size-5 rotate-90 md:rotate-0" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
