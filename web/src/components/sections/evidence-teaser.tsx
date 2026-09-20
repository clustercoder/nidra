import { ArrowRight } from "lucide-react";

const CHIPS = [
  { value: "45-dim", label: "host state" },
  { value: "K=6", label: "6-min horizon" },
  { value: "×5", label: "seed ensemble" },
  { value: "<300 ms", label: "CPU inference" },
];

/** Natural-prevalence AP against forecast horizon on the Thursday holdout (Run 8,
 *  ml/artifacts/metrics/holdout/benchmark.json, task_C_progression): the world
 *  model's forecast of "attack at t+k" against the same risk head applied to the
 *  TRUE future state (the oracle — an upper bound on what the head can recognise,
 *  not a competing forecaster). The forecast decays with horizon; the oracle is flat. */
function HorizonCurve() {
  const W = 630;
  const H = 404;
  const L = 58;
  const R = W - 24;
  const T = 44;
  const B = 300;

  const x = (k: number) => L + ((k - 1) / 5) * (R - L);
  const y = (v: number) => B - v * (B - T);

  const model = [0.582, 0.505, 0.454, 0.404, 0.352, 0.285];
  const oracle = [0.4, 0.393, 0.369, 0.436, 0.451, 0.416];

  const path = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${x(i + 1)} ${y(v)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-[630px] lg:ml-auto"
      role="img"
      aria-label="Line chart: average precision at natural prevalence against forecast horizon on the Thursday holdout day. The world model declines from 0.58 at one minute to 0.29 at six minutes; the oracle on the true future stays near 0.4."
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
        forecast horizon (60 s windows)
      </text>
      <text x={L} y={T - 20} className="fill-gray-dark" fontSize="11.5">
        AP (natural prevalence)
      </text>

      {/* oracle on the true future — dashed + round markers, distinguishable without colour */}
      <path d={path(oracle)} fill="none" className="stroke-gray-dark" strokeWidth="2" strokeDasharray="6 5" />
      {oracle.map((v, i) => (
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
          Oracle: same head on the true future
        </text>
      </g>
      <text x={L} y={H - 8} className="fill-gray-dark" fontSize="11" fontStyle="italic">
        Run 8, Thursday holdout, natural prevalence — test day: 0.063 → 0.024
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
              Ten baselines run alongside it on the same rows — persistence, the
              same head on the true future, a linear dynamics model, four
              classifiers — and every one it does not beat is published.
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
