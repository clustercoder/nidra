import { PipelineArt } from "@/components/art/pipeline-art";
import { StageLifecycle } from "@/components/art/stage-lifecycle";

const COLUMNS: {
  heading: string;
  intro: string;
  bullets: React.ReactNode[];
}[] = [
  {
    heading: "Ingest & Encode",
    intro: "Traffic becomes state:",
    bullets: [
      "Flow-level (NetFlow/IPFIX) and packet-level (tshark/PCAP) features",
      "45-dim state vector per host per 30-second window",
      "5 feature groups: flow, packet, graph, dynamics, activity",
    ],
  },
  {
    heading: "Learn the Dynamics",
    intro: "What the model learns:",
    bullets: [
      (
        <>
          2-layer GRU encoder + Gaussian transition model learns{" "}
          <span className="font-medium whitespace-nowrap">
            P(S<sub className="text-[0.7em]">t+1</sub> | S
            <sub className="text-[0.7em]">t</sub>)
          </span>
        </>
      ),
      "Trained self-supervised — no attack labels",
      "15-minute context window (L=30)",
    ],
  },
  {
    heading: "Simulate & Forecast",
    intro: "Roll the future forward:",
    bullets: [
      "Recursive rollout, K=6 steps (3-minute horizon)",
      "~500–1000 sampled trajectories via a 5-seed ensemble",
"Maps predicted state to one of six lifecycle stages",
    ],
  },
  {
    heading: "Explain & Alert",
    intro: "Every forecast comes with its reasons:",
    bullets: [
      "A risk score and the seconds of warning that go with it",
      "SHAP feature attribution + temporal saliency",
      "Flagged flows drill-down (flow bridge)",
    ],
  },
];

export function HowNidraWorks() {
  return (
    <section id="how-it-works" className="scroll-mt-20">
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="h-24 md:h-32 lg:h-44" />
        <h2 className="font-headings text-gray-black-soft text-center text-[1.375rem] leading-tight font-semibold md:text-[34px]">
          How NIDRA helps: From Traffic to a Forecast
        </h2>
        <p className="text-gray-dark mx-auto mt-5 max-w-[820px] text-center text-xl">
          Four steps. About 1000 sampled futures per host, recomputed every 30
          seconds.
        </p>
        <div className="mt-10 overflow-x-auto">
          <div className="min-w-[760px]">
            <PipelineArt className="mx-auto w-full" />
          </div>
        </div>
        <div className="mt-6 grid grid-cols-1 gap-10 md:grid-cols-2 lg:grid-cols-4">
          {COLUMNS.map((col) => (
            <div key={col.heading}>
              <h3 className="font-headings text-primary-blue text-lg/6 font-semibold uppercase md:text-[1.4rem] md:leading-tight">
                {col.heading}
              </h3>
              <p className="text-gray-black mt-4 text-base/6">{col.intro}</p>
              <ul className="text-gray-black mt-4 list-disc space-y-2.5 pl-5">
                {col.bullets.map((b, i) => (
                  <li key={i} className="text-base/6">
                    {b}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <StageLifecycle className="mt-14" />
        <div className="h-24 md:h-32 lg:h-44" />
      </div>
    </section>
  );
}
