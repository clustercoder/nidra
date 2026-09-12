import { ArrowRight } from "lucide-react";

const CHIPS = [
  { value: "45-dim", label: "host state" },
  { value: "K=6", label: "3-min horizon" },
  { value: "×5", label: "seed ensemble" },
  { value: "<300 ms", label: "CPU inference" },
];

/** AUC-PR against forecast horizon: the world model decays; persistence falls off a cliff. */
function HorizonCurve() {
  const W = 630;
  const H = 404;
  const L = 58;
  const R = W - 24;
  const T = 44;
  const B = 300;

  const x = (k: number) => L + ((k - 1) / 5) * (R - L);
  const y = (v: number) => B - v * (B - T);

  const model = [0.82, 0.79, 0.75, 0.72, 0.7, 0.68];
  const persistence = [0.79, 0.68, 0.58, 0.51, 0.47, 0.44];

  const path = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${x(i + 1)} ${y(v)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-[630px] lg:ml-auto"
      role="img"
      aria-label="Illustrative line chart: AUC-PR against forecast horizon. The world model declines slowly from k=1 to k=6; the persistence baseline declines faster."
    >
      {/* grid + axes */}
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={R} y1={y(v)} y2={y(v)} className="stroke-gray-light" strokeWidth="1" />
          <text x={L - 10} y={y(v) + 4} textAnchor="end" className="fill-gray-dark" fontSize="11">
            {v.toFixed(2)}
          </text>
        </g>
      ))}
      <line x1={L} x2={R} y1={B} y2={B} className="stroke-gray-medium" strokeWidth="1" />
      {[1, 2, 3, 4, 5, 6].map((k) => (
        <text key={k} x={x(k)} y={B + 20} textAnchor="middle" className="fill-gray-dark" fontSize="11">
          k={k}
        </text>
      ))}
      <text x={(L + R) / 2} y={B + 42} textAnchor="middle" className="fill-gray-dark" fontSize="11.5">
        forecast horizon (30 s windows)
      </text>
      <text x={L} y={T - 20} className="fill-gray-dark" fontSize="11.5">
        AUC-PR
      </text>

      {/* persistence baseline — dashed + round markers, distinguishable without colour */}
      <path d={path(persistence)} fill="none" className="stroke-gray-dark" strokeWidth="2" strokeDasharray="6 5" />
      {persistence.map((v, i) => (
        <circle key={i} cx={x(i + 1)} cy={y(v)} r="3.5" className="fill-gray-dark" />
      ))}

      {/* world model — solid + square markers */}
      <path d={path(model)} fill="none" className="stroke-observed" strokeWidth="2.5" />
      {model.map((v, i) => (
        <rect key={i} x={x(i + 1) - 3.5} y={y(v) - 3.5} width="7" height="7" className="fill-observed" />
      ))}

      {/* legend */}
      <g transform={`translate(${L} ${B + 64})`}>
        <rect x="0" y="-5" width="9" height="9" className="fill-observed" />
        <text x="16" y="3" className="fill-gray-black-soft" fontSize="12">
          World model
        </text>
        <circle cx="126" cy="0" r="4" className="fill-gray-dark" />
        <text x="138" y="3" className="fill-gray-black-soft" fontSize="12">
          Persistence baseline
        </text>
      </g>
      <text x={L} y={H - 8} className="fill-gray-dark" fontSize="11" fontStyle="italic">
        illustrative — pending artifacts/metrics
      </text>
    </svg>
  );
}

export function EvidenceTeaser() {
  return (
    <section className="relative overflow-hidden bg-white py-24 lg:py-32">
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="grid grid-cols-1 items-center gap-10 lg:grid-cols-2">
          {/* Left */}
          <div className="space-y-5">
            <p className="text-primary-blue text-base font-semibold tracking-wide uppercase">
              How we check
            </p>
            <h2 className="font-headings text-gray-black-soft text-[2rem] leading-tight font-semibold lg:text-[40px]">
              Does it beat guessing?
            </h2>
            <div className="flex flex-wrap items-stretch gap-2">
              {CHIPS.map((chip) => (
                <div
                  key={chip.value}
                  className="border-gray-light rounded-lg border px-4 py-3"
                >
                  <div className="font-headings text-gray-black-soft text-xl font-semibold">
                    {chip.value}
                  </div>
                  <div className="text-gray-dark text-sm">{chip.label}</div>
                </div>
              ))}
            </div>
            <p className="text-gray-dark text-base">
              Two baselines run alongside it: persistence, and the same model with
              the context shuffled.
            </p>
            <a
              href="#evidence"
              className="group/button bg-primary-blue relative inline-flex max-w-fit items-center gap-1.5 rounded-full border border-transparent px-[22px] py-3 text-base font-medium text-white transition-all duration-200 hover:scale-[1.03] hover:bg-[rgb(24,64,180)]"
            >
              See the numbers
              <ArrowRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
            </a>
          </div>
          {/* Right */}
          <div>
            <HorizonCurve />
          </div>
        </div>
      </div>
    </section>
  );
}
