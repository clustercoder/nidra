"use client";

import { useState } from "react";
import { ConsoleMock, type ConsoleVariant } from "@/components/console-mock";

const TABS: { label: string; desc: string; variant: ConsoleVariant }[] = [
  {
    label: "Forecast",
    desc: "The risk curve per host, and the seconds of warning on it.",
    variant: "forecast",
  },
  {
    label: "Attack Stage",
    desc: "Where each host sits across the six stages.",
    variant: "stage",
  },
  {
    label: "Explainability",
    desc: "Which of the 45 features moved the projection, and when.",
    variant: "explain",
  },
  {
    label: "Model Internals",
    desc: "Ablations and calibration. The view a judge would ask for.",
    variant: "internals",
  },
];

export function ForecastConsole() {
  const [active, setActive] = useState(0);

  return (
    <section className="relative isolate overflow-hidden bg-white">
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="h-24 md:h-32 lg:h-44" />
        {/* header */}
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-[420px_1fr] lg:gap-16">
          <h2 className="font-headings text-primary-blue text-[2rem] leading-tight font-semibold lg:text-[46px]">
            What you actually look at
          </h2>
          <div className="space-y-6">
            <p className="text-gray-black max-w-2xl text-lg leading-relaxed">
              Four views on the same forecast: the curve, the stage, the reasons
              behind it, and the numbers we are judged on.
            </p>
            <p className="text-gray-dark text-base font-semibold tracking-wide">
              CIC-IDS2017 · tshark · PyTorch
            </p>
          </div>
        </div>
        <div className="h-10 lg:h-14" />
        {/* tabs + pane */}
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-[300px_1fr] lg:gap-14">
          <div className="divide-gray-light border-gray-light h-fit divide-y rounded-lg border">
            {TABS.map((t, i) => (
              <button
                key={t.label}
                onClick={() => setActive(i)}
                aria-pressed={i === active}
                className={`hover:bg-gray-light/50 relative flex w-full cursor-pointer items-center px-6 py-5 text-left text-lg font-medium transition-colors ${
                  i === active
                    ? "bg-gray-light/40 text-primary-blue"
                    : "text-gray-dark"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div>
            <h3 className="font-headings text-gray-black-soft text-2xl font-semibold">
              {TABS[active].label}
            </h3>
            <p className="text-gray-dark mt-2 text-xl">{TABS[active].desc}</p>
            <div className="mt-5">
              <ConsoleMock variant={TABS[active].variant} />
            </div>
            <p className="text-gray-dark mt-3 text-sm">
              Console mock with illustrative values — the live view ships with
              the replay demo.
            </p>
            <ul className="mt-5 flex items-center justify-center gap-3">
              {TABS.map((t, i) => (
                <li key={t.label}>
                  <button
                    aria-label={`Show ${t.label}`}
                    onClick={() => setActive(i)}
                    className={`size-2 rounded-full ${
                      i === active ? "bg-primary-blue" : "bg-gray-light"
                    }`}
                  />
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="h-24 md:h-32 lg:h-44" />
      </div>
    </section>
  );
}
