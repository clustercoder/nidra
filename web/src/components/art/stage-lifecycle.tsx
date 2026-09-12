import {
  ChevronRight,
  Database,
  Lock,
  Radio,
  Share2,
  ShieldCheck,
  Telescope,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

/**
 * The six stages a predicted state can map to, as a strip rather than an
 * arrow-chain sentence. Colour runs cool to warm across the lifecycle, but the
 * label carries the meaning on its own.
 */
const STAGES: { label: string; Icon: LucideIcon; ring: string }[] = [
  { label: "Benign", Icon: ShieldCheck, ring: "bg-node-mint/60 border-positive text-positive" },
  { label: "Recon", Icon: Telescope, ring: "bg-node-blue/60 border-observed text-observed" },
  {
    label: "Initial Access",
    Icon: Lock,
    ring: "bg-threshold/15 border-threshold text-threshold",
  },
  { label: "Lateral", Icon: Share2, ring: "bg-negative/12 border-negative text-negative" },
  { label: "C2", Icon: Radio, ring: "bg-node-lavender/60 border-projected text-projected" },
  { label: "Exfil", Icon: Database, ring: "bg-node-lavender/60 border-projected text-projected" },
];

export function StageLifecycle({ className = "" }: { className?: string }) {
  return (
    <div className={`border-gray-light bg-gray-off-white mx-auto w-full max-w-4xl min-w-0 overflow-x-auto rounded-2xl border p-6 ${className}`}>
      <div className="flex min-w-[640px] items-start justify-center gap-x-2">
        {STAGES.map(({ label, Icon, ring }, i) => (
          <div key={label} className="contents">
            <div className="flex w-[92px] shrink-0 flex-col items-center text-center">
              <span
                className={`flex size-14 items-center justify-center rounded-full border-2 ${ring}`}
              >
                <Icon className="size-6" strokeWidth={1.9} />
              </span>
              <span className="text-gray-black-soft mt-2.5 text-sm leading-tight">
                {label}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <ChevronRight
                className="text-gray-medium mt-5 size-5 shrink-0"
                aria-hidden
              />
            )}
          </div>
        ))}
      </div>
      <p className="text-gray-dark mt-4 text-center text-xs tracking-[0.12em] uppercase">
        6-stage attack lifecycle
      </p>
    </div>
  );
}
