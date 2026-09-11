import { Sparkle } from "lucide-react";

export type ConsoleVariant = "forecast" | "stage" | "explain" | "internals";

/** Mock console data — illustrative, consistent across variants. */
const STAGES = [
  { label: "benign", count: 17, cls: "bg-positive" },
  { label: "recon", count: 3, cls: "bg-observed" },
  { label: "initial_access", count: 2, cls: "bg-projected" },
  { label: "lateral", count: 1, cls: "bg-threshold" },
  { label: "c2", count: 1, cls: "bg-negative" },
  { label: "exfil", count: 0, cls: "bg-gray-medium" },
];

const TOTAL = STAGES.reduce((sum, s) => sum + s.count, 0);

const DONUT_COLORS = [
  "var(--positive)",
  "var(--observed)",
  "var(--projected)",
  "var(--threshold)",
  "var(--negative)",
  "rgb(203 213 225)",
];

function donutGradient() {
  let acc = 0;
  const stops = STAGES.map((s, i) => {
    const from = (acc / TOTAL) * 100;
    acc += s.count;
    const to = (acc / TOTAL) * 100;
    return `${DONUT_COLORS[i]} ${from.toFixed(2)}% ${to.toFixed(2)}%`;
  });
  return `conic-gradient(${stops.join(", ")})`;
}

const HOSTS = [
  { ip: "192.168.10.14", risk: 0.82, stage: "lateral", lead: "90 s" },
  { ip: "192.168.10.51", risk: 0.77, stage: "c2", lead: "60 s" },
  { ip: "192.168.10.8", risk: 0.64, stage: "initial_access", lead: "—" },
  { ip: "192.168.10.25", risk: 0.41, stage: "recon", lead: "—" },
  { ip: "192.168.10.3", risk: 0.18, stage: "benign", lead: "—" },
];

const DRIVERS = [
  { feature: "unique_dst_ports", value: 0.31 },
  { feature: "flow_count_delta", value: 0.24 },
  { feature: "syn_ratio", value: 0.19 },
  { feature: "out_in_byte_ratio", value: -0.12 },
  { feature: "mean_iat_slope", value: -0.08 },
];

const ABLATIONS = [
  { label: "World model vs persistence — AUC-PR", a: 0.68, b: 0.44 },
  { label: "Time-shuffle control", a: 0.68, b: 0.21 },
];

function CardShell({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-gray-light flex flex-col rounded-lg border bg-white">
      <div className="border-gray-light text-gray-black-soft border-b px-4 py-3 text-sm font-semibold">
        {title}
      </div>
      <div className="flex-1 p-4">{children}</div>
    </div>
  );
}

function StageDonut({ emphasised }: { emphasised?: boolean }) {
  return (
    <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
      <div
        className={`relative shrink-0 rounded-full ${emphasised ? "size-40" : "size-32"}`}
        style={{ background: donutGradient() }}
        role="img"
        aria-label={`24 hosts monitored: ${STAGES.map((s) => `${s.count} ${s.label}`).join(", ")}`}
      >
        <div className="absolute inset-[22%] flex flex-col items-center justify-center rounded-full bg-white">
          <span className="font-headings text-gray-black-soft text-xl font-semibold">
            {TOTAL}
          </span>
          <span className="text-gray-dark px-1 text-center text-[9px] leading-[1.15]">
            Hosts Monitored
          </span>
        </div>
      </div>
      <ul className="flex-1 space-y-1.5">
        {STAGES.map((s) => (
          <li key={s.label} className="flex items-center gap-2 text-xs">
            <span className={`size-2.5 shrink-0 rounded-sm ${s.cls}`} aria-hidden />
            <span className="text-gray-black-soft flex-1">{s.label}</span>
            <span className="text-gray-dark tabular-nums">{s.count}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function HostTable() {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[320px] text-left text-xs">
        <thead>
          <tr className="text-gray-dark">
            <th className="pb-2 font-medium">Host</th>
            <th className="pb-2 font-medium">Risk</th>
            <th className="pb-2 font-medium">Predicted stage</th>
            <th className="pb-2 text-right font-medium">Lead time</th>
          </tr>
        </thead>
        <tbody className="text-gray-black-soft">
          {HOSTS.map((h) => (
            <tr key={h.ip} className="border-gray-light border-t">
              <td className="py-2 font-mono text-[11px]">{h.ip}</td>
              <td className="py-2">
                <span className="flex items-center gap-2">
                  <span className="tabular-nums">{h.risk.toFixed(2)}</span>
                  <span className="bg-gray-light h-1.5 w-12 overflow-hidden rounded-full">
                    <span
                      className={`block h-full rounded-full ${h.risk >= 0.75 ? "bg-threshold" : "bg-observed"}`}
                      style={{ width: `${h.risk * 100}%` }}
                    />
                  </span>
                </span>
              </td>
              <td className="py-2">{h.stage}</td>
              <td className="py-2 text-right tabular-nums">{h.lead}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DriverList() {
  return (
    <ul className="space-y-3">
      {DRIVERS.map((d) => (
        <li key={d.feature} className="text-xs">
          <div className="text-gray-black-soft flex items-center justify-between">
            <span className="font-mono text-[11px]">{d.feature}</span>
            <span className="tabular-nums">
              {d.value > 0 ? "+" : ""}
              {d.value.toFixed(2)}
            </span>
          </div>
          <div className="bg-gray-light mt-1.5 flex h-1.5 overflow-hidden rounded-full">
            <span className="flex w-1/2 justify-end">
              {d.value < 0 && (
                <span
                  className="bg-negative block h-full rounded-full"
                  style={{ width: `${Math.abs(d.value) * 200}%` }}
                />
              )}
            </span>
            <span className="flex w-1/2">
              {d.value > 0 && (
                <span
                  className="bg-positive block h-full rounded-full"
                  style={{ width: `${d.value * 200}%` }}
                />
              )}
            </span>
          </div>
        </li>
      ))}
    </ul>
  );
}

function AblationList() {
  return (
    <div className="space-y-4">
      {ABLATIONS.map((a) => (
        <div key={a.label} className="text-xs">
          <p className="text-gray-black-soft">{a.label}</p>
          <div className="mt-2 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-gray-dark w-20 shrink-0">world model</span>
              <span className="bg-gray-light h-2 flex-1 overflow-hidden rounded-full">
                <span
                  className="bg-observed block h-full rounded-full"
                  style={{ width: `${a.a * 100}%` }}
                />
              </span>
              <span className="text-gray-black-soft w-8 text-right tabular-nums">
                {a.a.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-dark w-20 shrink-0">baseline</span>
              <span className="bg-gray-light h-2 flex-1 overflow-hidden rounded-full">
                <span
                  className="bg-gray-dark block h-full rounded-full"
                  style={{ width: `${a.b * 100}%` }}
                />
              </span>
              <span className="text-gray-black-soft w-8 text-right tabular-nums">
                {a.b.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      ))}
      <p className="text-gray-dark text-[11px] italic">
        illustrative until artifacts/metrics is populated
      </p>
    </div>
  );
}

function StageTimeline() {
  return (
    <div className="mt-5">
      <p className="text-gray-dark mb-2 text-[11px] tracking-wide uppercase">
        Predicted stage over the horizon · 192.168.10.14
      </p>
      <div className="flex gap-1">
        {["benign", "recon", "recon", "initial_access", "lateral", "lateral"].map(
          (stage, i) => (
            <div
              key={i}
              className="border-gray-light flex-1 rounded border px-1 py-1.5 text-center"
            >
              <div className="text-gray-black-soft truncate text-[9px]">
                {stage}
              </div>
              <div className="text-gray-dark text-[9px]">k={i + 1}</div>
            </div>
          ),
        )}
      </div>
    </div>
  );
}

const RIGHT_TITLE: Record<ConsoleVariant, string> = {
  forecast: "Highest-Risk Hosts",
  stage: "Highest-Risk Hosts",
  explain: "Top Risk Drivers",
  internals: "Ablations",
};

export function ConsoleMock({ variant }: { variant: ConsoleVariant }) {
  return (
    <div className="border-primary-blue overflow-hidden rounded-xl border bg-white">
      <div className="bg-primary-blue flex h-8 items-center gap-2 px-4">
        <Sparkle className="size-3.5 fill-white text-white" />
        <span className="text-[11px] text-white/80">
          NIDRA console · replay fixture
        </span>
      </div>
      <div className="bg-gray-off-white grid gap-4 p-4 lg:grid-cols-2">
        <CardShell title="Breakdown by Stage">
          <StageDonut emphasised={variant === "stage"} />
          {variant === "stage" && <StageTimeline />}
        </CardShell>
        <CardShell title={RIGHT_TITLE[variant]}>
          {variant === "explain" ? (
            <DriverList />
          ) : variant === "internals" ? (
            <AblationList />
          ) : (
            <HostTable />
          )}
        </CardShell>
      </div>
    </div>
  );
}
