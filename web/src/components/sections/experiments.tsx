"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronRight } from "lucide-react";
import { GITHUB_URL } from "@/components/nav-data";

const METRICS_URL = `${GITHUB_URL}/tree/main/artifacts/metrics`;

const EXPERIMENTS = [
  {
    kind: "Ablation",
    title: "Persistence baseline",
    subtitle: "Can the model beat “the next window looks like this one”?",
    blurb:
      "If persistence keeps up, the model is not earning its complexity. That result gets published and diagnosed rather than buried.",
    chart: true,
  },
  {
    kind: "Ablation",
    title: "Time-shuffle control",
    subtitle: "Does temporal order actually matter?",
    blurb:
      "Shuffle the 30 context windows and the forecast should fall apart. If it survives, the model was never reading time.",
    chart: false,
  },
  {
    kind: "Holdout",
    title: "Thursday holdout",
    subtitle: "Infiltration — an attack type never seen in training.",
    blurb:
      "Thursday never appears in training. Whatever the model scores there, it scored without having seen the attack.",
    chart: false,
  },
];

const PRINCIPLES = [
  {
    heading: "Falsifiable",
    body: "Heads train on observed states only, then freeze. Nothing forward-looking can come from anywhere else.",
  },
  {
    heading: "Leak-free",
    body: "Nothing after time t reaches the input. Normalisation is fitted on the training period and serialised.",
  },
  {
    heading: "Reproducible",
    body: "Pinned seeds, one config file, and the metrics committed next to the code.",
  },
];

const STATS = [
  { value: 45, prefix: "", suffix: "", label: "features per host state" },
  { value: 180, prefix: "", suffix: "s", label: "seconds of forecast horizon" },
  { value: 300, prefix: "<", suffix: "", label: "ms max CPU inference" },
];

function CountUp({
  target,
  prefix = "",
  suffix = "",
}: {
  target: number;
  prefix?: string;
  suffix?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  // Start at the true value. These are load-bearing constants, so the served
  // markup must state them correctly even if the animation never runs — a tile
  // reading "0 features per host state" is a false claim, not a missing effect.
  const [value, setValue] = useState(target);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Only count up when the tile starts below the fold; otherwise leave the
    // number alone rather than flashing it back to zero under the reader.
    if (el.getBoundingClientRect().top < window.innerHeight) return;
    setValue(0);
    const obs = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) return;
        obs.disconnect();
        const start = performance.now();
        const duration = 1500;
        const tick = (now: number) => {
          const p = Math.min((now - start) / duration, 1);
          setValue(Math.round(target * (1 - Math.pow(1 - p, 3))));
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.5 },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [target]);

  return (
    <span ref={ref}>
      {prefix}
      {value}
      {suffix}
    </span>
  );
}

/** World model vs persistence, AUC-PR. Placeholder shape, labelled as such. */
function AblationBars() {
  const bars = [
    { label: "World model", value: 0.68, cls: "fill-observed" },
    { label: "Persistence", value: 0.44, cls: "fill-gray-dark" },
  ];
  const W = 400;
  const H = 260;
  const base = 200;
  const top = 40;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-[400px]"
      role="img"
      aria-label="Illustrative bar chart comparing AUC-PR of the world model against the persistence baseline."
    >
      <line
        x1="60"
        x2={W - 20}
        y1={base}
        y2={base}
        className="stroke-gray-medium"
        strokeWidth="1"
      />
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line
            x1="60"
            x2={W - 20}
            y1={base - v * (base - top)}
            y2={base - v * (base - top)}
            className="stroke-gray-light"
            strokeWidth="1"
          />
          <text
            x="52"
            y={base - v * (base - top) + 4}
            textAnchor="end"
            className="fill-gray-dark"
            fontSize="11"
          >
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {bars.map((b, i) => {
        const x = 100 + i * 140;
        const h = b.value * (base - top);
        return (
          <g key={b.label}>
            <rect
              x={x}
              y={base - h}
              width="80"
              height={h}
              rx="4"
              className={b.cls}
            />
            <text
              x={x + 40}
              y={base - h - 8}
              textAnchor="middle"
              className="fill-gray-black-soft"
              fontSize="12"
              fontWeight="600"
            >
              {b.value.toFixed(2)}
            </text>
            <text
              x={x + 40}
              y={base + 18}
              textAnchor="middle"
              className="fill-gray-dark"
              fontSize="12"
            >
              {b.label}
            </text>
          </g>
        );
      })}
      <text x="60" y="22" className="fill-gray-dark" fontSize="11">
        AUC-PR at K=6
      </text>
      <text x="60" y={H - 12} className="fill-gray-dark" fontSize="11" fontStyle="italic">
        illustrative — pending artifacts/metrics
      </text>
    </svg>
  );
}

export function Experiments() {
  const [active, setActive] = useState(0);
  const experiment = EXPERIMENTS[active];

  return (
    <section id="evidence" className="scroll-mt-20">
      <div className="h-24 md:h-32 lg:h-44" />
      {/* the experiments carousel */}
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <h2 className="font-headings text-gray-black-soft text-center text-[1.375rem] leading-tight font-semibold md:text-[34px]">
          Three things that could{" "}
          <span className="text-primary-blue">prove us wrong</span>
        </h2>
        <div className="h-10 md:h-14" />
        <div className="border-primary-pink rounded-2xl border bg-white p-8 shadow-[0_20px_60px_-30px_rgba(29,78,216,0.25)] md:p-12">
          <div className="grid items-center gap-10 lg:grid-cols-2">
            <div className="space-y-4">
              <div className="text-gray-dark text-base font-semibold tracking-[0.18em] uppercase">
                {experiment.kind}
              </div>
              <h3 className="font-headings text-gray-black-soft text-[26px] font-semibold">
                {experiment.title}
              </h3>
              <p className="text-gray-black text-xl">{experiment.subtitle}</p>
              <p className="text-gray-black text-lg leading-relaxed">
                {experiment.blurb}
              </p>
              <a
                href={METRICS_URL}
                className="group/button bg-primary-blue inline-flex max-w-fit items-center gap-1.5 rounded-full px-[22px] py-3 text-base font-medium text-white transition-all duration-200 hover:scale-[1.03] hover:bg-[rgb(24,64,180)]"
              >
                See metrics
                <ChevronRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
              </a>
            </div>
            <div className="flex justify-center">
              {experiment.chart ? (
                <AblationBars />
              ) : (
                <div className="border-gray-light text-gray-dark w-full max-w-[400px] rounded-xl border border-dashed p-10 text-center text-base">
                  Metrics for this experiment land in
                  <span className="font-mono"> artifacts/metrics/ </span>
                  once the evaluation run completes.
                </div>
              )}
            </div>
          </div>
        </div>
        <ul className="mt-5 flex items-center justify-center gap-2">
          {EXPERIMENTS.map((e, i) => (
            <li key={e.title}>
              <button
                aria-label={`go to experiment ${i + 1}`}
                onClick={() => setActive(i)}
                className={`size-2 rounded-full ${
                  i === active ? "bg-primary-blue" : "bg-gray-light"
                }`}
              />
            </li>
          ))}
        </ul>
      </div>
      <div className="h-24 md:h-32 lg:h-44" />
      {/* key numbers band */}
      <div className="bg-primary-blue relative overflow-hidden py-10">
        <div className="to-secondary-blue-medium/60 absolute inset-y-0 right-0 w-1/2 bg-linear-to-r from-transparent" />
        <div className="relative mx-auto flex max-w-4xl flex-col items-center gap-5">
          <div className="flex flex-col items-stretch justify-center gap-4 px-6 md:flex-row">
            <div className="flex items-center gap-5 rounded-lg bg-white px-6 py-5">
              <span className="font-headings text-primary-blue text-3xl font-semibold">
                3 min
              </span>
              <div className="text-gray-dark text-base">
                <div className="text-gray-black-soft font-semibold">
                  forecast horizon
                </div>
                K=6 windows
              </div>
            </div>
            <div className="flex items-center gap-5 rounded-lg bg-white px-6 py-5">
              <span className="font-headings text-primary-blue text-3xl font-semibold">
                30 s
              </span>
              <div className="text-gray-dark text-base">
                <div className="text-gray-black-soft font-semibold">
                  state resolution
                </div>
                per host
              </div>
            </div>
          </div>
          <a
            href={METRICS_URL}
            className="inline-flex items-center gap-1.5 font-medium text-white transition-opacity hover:opacity-80"
          >
            How we evaluate
            <ChevronRight className="size-4" />
          </a>
        </div>
      </div>
      <div className="h-24 md:h-32 lg:h-44" />
      {/* constants + principles */}
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="relative">
          <h2 className="font-headings text-gray-black-soft text-[2rem] font-semibold lg:text-[42px]">
            Fixed by design
          </h2>
          <div className="mt-12 grid grid-cols-1 gap-14 md:grid-cols-2 md:gap-8 lg:grid-cols-3">
            {STATS.map((stat, i) => (
              <div key={stat.label} className="flex flex-col gap-4">
                <div className="gradient-pink-br text-primary-blue flex items-center gap-2 px-2 md:px-4">
                  <span className="font-headings text-[52px] leading-[60px] font-bold">
                    <CountUp
                      target={stat.value}
                      prefix={stat.prefix}
                      suffix={stat.suffix}
                    />
                  </span>
                  <span className="text-sm">{stat.label}</span>
                </div>
                <div className="flex-1 md:pt-4">
                  <h3 className="font-headings text-gray-black-soft text-2xl font-semibold md:text-[1.625rem]/9">
                    {PRINCIPLES[i].heading}
                  </h3>
                  <p className="text-gray-black py-4 text-base/6">
                    {PRINCIPLES[i].body}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="h-24 md:h-32 lg:h-44" />
    </section>
  );
}
