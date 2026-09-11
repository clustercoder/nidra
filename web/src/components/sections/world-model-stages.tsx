"use client";

import { useState } from "react";
import {
  ArrowRight,
  BrainCircuit,
  ChartBar,
  Eye,
  FileText,
  FlaskConical,
  Gauge,
  ListTree,
  TrendingUp,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { StageDiagram, type DiagramNode } from "@/components/stage-diagram";

type TabKey = "observe" | "forecast" | "explain";

interface Tab {
  key: TabKey;
  label: string;
  Icon: LucideIcon;
  eyebrow: string;
  heading: string;
  body: string;
  subHeading: string;
  subBody: string;
  diagramTitle: string;
  nodes: DiagramNode[];
}

const TABS: Tab[] = [
  {
    key: "observe",
    label: "Observe",
    Icon: Eye,
    eyebrow: "OBSERVE",
    heading: "From raw traffic to host state",
    body: "We capture live network flows, aggregate them per host in 30-second windows, and convert the activity into a 45-dimensional state vector.",
    subHeading: "Why per-host, not per-flow?",
    subBody:
      "A compromise unfolds over time on a machine — scanning, probing, foothold, lateral movement, exfiltration. Per-flow data shows only a slice. Per-host sequences reveal the trajectory.",
    diagramTitle: "The Observe stage",
    nodes: [
      {
        icon: "tshark",
        label: "Live network flows",
        sublabel: "pcap · flow data",
        color: "blue",
      },
      {
        icon: "30 s",
        label: "30-second window",
        sublabel: "per host",
        color: "lavender",
      },
      {
        icon: "45-d",
        label: "45-dim state vector",
        sublabel: "per host",
        color: "mint",
      },
    ],
  },
  {
    key: "forecast",
    label: "Forecast",
    Icon: TrendingUp,
    eyebrow: "FORECAST",
    heading: "From state to a range of futures",
    body: "A 2-layer GRU encoder and a Gaussian transition model learn how host state evolves — trained self-supervised on transitions, no attack labels needed. At inference, NIDRA rolls the state forward K=6 steps and samples ~500–1000 trajectories across a 5-seed ensemble.",
    subHeading: "Why sample at all?",
    subBody:
      "A single projection tells you nothing about how sure the model is. Sampling many gives you a band, and the band is what makes the number usable.",
    diagramTitle: "The Forecast stage",
    nodes: [
      {
        icon: "L=30",
        label: "15-min context",
        sublabel: "30 windows",
        color: "blue",
      },
      {
        icon: BrainCircuit,
        label: "Learned dynamics",
        sublabel: "GRU + Gaussian transition",
        color: "lavender",
      },
      {
        icon: "K=6",
        label: "3-min rollout",
        sublabel: "~1000 trajectories",
        color: "mint",
      },
    ],
  },
  {
    key: "explain",
    label: "Explain",
    Icon: FileText,
    eyebrow: "EXPLAIN",
    heading: "From forecast to action",
    body: "Each forecast comes with the reasons behind it. SHAP attribution over the 45 features, saliency across the context window, and the specific flows the projection was built from.",
    subHeading: "Falsifiable by design",
    subBody:
      "The risk and stage heads are trained on observed states only, then frozen. Every unit of forward-looking capability must come from the transition model.",
    diagramTitle: "The Explain stage",
    nodes: [
      {
        icon: Gauge,
        label: "Risk + lead time",
        sublabel: "threshold 0.75",
        color: "blue",
      },
      {
        icon: ChartBar,
        label: "SHAP + saliency",
        sublabel: "feature attribution",
        color: "lavender",
      },
      {
        icon: ListTree,
        label: "Flagged flows",
        sublabel: "drill-down",
        color: "mint",
      },
    ],
  },
];

export function WorldModelStages() {
  const [active, setActive] = useState<TabKey>("observe");
  const tab = TABS.find((t) => t.key === active)!;

  return (
    <section className="from-primary-pink/40 via-gray-light/40 to-gray-light/0 bg-linear-to-t">
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="h-24 md:h-32 lg:h-44" />
        {/* header */}
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:gap-20">
          <div className="space-y-8">
            <h2 className="font-headings text-gray-black-soft text-[1.75rem] leading-tight font-semibold lg:text-[42px]">
              One model, three jobs
            </h2>
            <a
              href="#how-it-works"
              className="group/button bg-primary-blue inline-flex max-w-fit items-center gap-1.5 rounded-full px-[22px] py-3 text-base font-medium text-white transition-all duration-200 hover:scale-[1.03] hover:bg-[rgb(24,64,180)]"
            >
              Explore how it works
              <ArrowRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
            </a>
          </div>
          <div className="text-gray-black space-y-4 text-xl leading-relaxed">
            <p className="text-gray-black-soft font-semibold">
              What the model actually does
            </p>
            <p>
              Every 30 seconds it reads each host&apos;s traffic into a state
              vector, runs that state forward, and shows which features moved
              the projected risk. All of it on CPU.
            </p>
            <p className="text-primary-blue font-semibold">
              The output is a number of seconds, not a verdict on something that
              already happened.
            </p>
          </div>
        </div>
        <div className="h-10 lg:h-16" />
        {/* tab bar */}
        <div className="flex flex-wrap items-center gap-4 md:gap-8">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setActive(t.key)}
              aria-pressed={active === t.key}
              className={`group flex items-center gap-2 text-xl font-semibold transition-all duration-100 ease-in ${
                active === t.key
                  ? "text-gray-black-soft"
                  : "text-gray-dark opacity-40 hover:opacity-70"
              }`}
            >
              <span
                className={`flex size-9 items-center justify-center rounded-full ${
                  active === t.key
                    ? "bg-primary-blue/10 text-primary-blue"
                    : "bg-gray-light text-gray-dark"
                }`}
              >
                <t.Icon className="size-5" strokeWidth={1.75} />
              </span>
              {t.label}
            </button>
          ))}
        </div>
        <div className="h-4" />
        {/* card */}
        <div className="border-primary-blue rounded-xl border bg-white p-6 md:p-8 md:px-14">
          <div className="grid lg:grid-cols-[300px_1fr] lg:gap-8">
            <div className="space-y-4">
              <span className="bg-primary-blue/10 text-primary-blue flex size-12 items-center justify-center rounded-full">
                <tab.Icon className="size-6" strokeWidth={1.75} />
              </span>
              <p className="text-primary-blue text-base font-semibold tracking-wide uppercase">
                {tab.eyebrow}
              </p>
              <h3 className="font-headings text-gray-black-soft text-[28px] font-semibold">
                {tab.heading}
              </h3>
              <p className="text-gray-black text-lg leading-relaxed">
                {tab.body}
              </p>
              <div className="border-gray-light border-t pt-4">
                <p className="text-gray-black-soft text-base font-semibold">
                  {tab.subHeading}
                </p>
                <p className="text-gray-dark mt-2 text-base leading-relaxed">
                  {tab.subBody}
                </p>
              </div>
            </div>
            <div className="mt-8 lg:mt-0">
              <StageDiagram title={tab.diagramTitle} nodes={tab.nodes} />
            </div>
          </div>
        </div>
        <div className="h-6" />
        {/* callout bar */}
        <div className="border-gray-light grid gap-6 rounded-xl border bg-white p-6 md:grid-cols-2 md:gap-10 md:p-8">
          <div className="flex gap-4">
            <span className="bg-node-blue/50 text-observed flex size-10 shrink-0 items-center justify-center rounded-lg">
              <ChartBar className="size-5" strokeWidth={1.75} />
            </span>
            <div>
              <p className="text-gray-black-soft text-lg font-semibold">
                Model input: 45-dimensional host state vector
              </p>
              <p className="text-gray-dark mt-2 text-base leading-relaxed">
                The 45 features capture flow statistics, temporal patterns and
                protocol behaviours, giving the model a complete picture of what
                the host is doing in that 30-second window.
              </p>
            </div>
          </div>
          <div className="flex gap-4 md:border-gray-light md:border-l md:pl-10">
            <span className="bg-node-lavender/50 text-projected flex size-10 shrink-0 items-center justify-center rounded-lg">
              <FlaskConical className="size-5" strokeWidth={1.75} />
            </span>
            <div>
              <p className="text-gray-black-soft text-base font-semibold">
                What-if simulation
              </p>
              <p className="text-gray-dark mt-2 text-base leading-relaxed">
                The model can also simulate alternative futures (e.g., &ldquo;what
                if the host starts scanning?&rdquo;) to show how projected risk
                changes. This is a model-internal what-if — a question about the
                model, not the network.
              </p>
            </div>
          </div>
        </div>
        <div className="h-24 md:h-32 lg:h-44" />
      </div>
    </section>
  );
}
