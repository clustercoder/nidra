import { Container, Database, Flame, Layers, Network, Zap } from "lucide-react";
import type { LucideIcon } from "lucide-react";

const STACK: { label: string; Icon: LucideIcon }[] = [
  { label: "CIC-IDS2017", Icon: Database },
  { label: "tshark", Icon: Network },
  { label: "PyTorch", Icon: Flame },
  { label: "FastAPI", Icon: Zap },
  { label: "Redis", Icon: Layers },
  { label: "Docker", Icon: Container },
];

export function StackRow() {
  return (
    <section className="overflow-hidden bg-white">
      <div className="h-12 md:h-16 lg:h-24" />
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <p className="text-gray-dark mx-auto max-w-[900px] text-lg text-center md:text-xl/8 lg:text-2xl">
          Built in the open, on public data
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-10">
          {STACK.map(({ label, Icon }) => (
            <div key={label} className="flex items-center gap-2.5">
              <Icon className="text-gray-dark/70 size-5" strokeWidth={1.75} />
              <span className="font-headings text-gray-dark/80 text-xl font-semibold">
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="h-12 md:h-16 lg:h-24" />
    </section>
  );
}
