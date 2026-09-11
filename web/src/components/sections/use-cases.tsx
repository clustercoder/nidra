"use client";

import { useRef } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Radar,
  ServerOff,
  Shuffle,
  Timer,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { CardArt, type CardScene } from "@/components/art/card-art";

const CARDS: {
  Icon: LucideIcon;
  title: string;
  body: string;
  tint: string;
  scene: CardScene;
}[] = [
  {
    Icon: Radar,
    title: "SOC early warning",
    body: "When the curve crosses 0.75 before anything has happened, triage becomes something you schedule rather than something that wakes you up.",
    tint: "from-gray-black to-[rgb(20,44,78)]",
    scene: "warning",
  },
  {
    Icon: Timer,
    title: "Incident response head start",
    body: "The forecast arrives with the flows it was built from, so the first thing a responder opens is the traffic that caused it.",
    tint: "from-gray-black to-[rgb(43,36,72)]",
    scene: "headstart",
  },
  {
    Icon: ServerOff,
    title: "Air-gapped deployment",
    body: "Passive tap, CPU inference, nothing phones home. It runs where the traffic already is.",
    tint: "from-gray-black to-[rgb(22,52,44)]",
    scene: "airgap",
  },
  {
    Icon: Shuffle,
    title: "Unseen attack patterns",
    body: "Thursday's Infiltration traffic is held out of training entirely, so the score on it is a real test of generalisation.",
    tint: "from-gray-black to-[rgb(60,42,20)]",
    scene: "unseen",
  },
  {
    Icon: Crosshair,
    title: "Red-team replay",
    body: "Replay an exercise at 60× and see which moves the model caught early, and which it missed.",
    tint: "from-gray-black to-[rgb(56,26,38)]",
    scene: "replay",
  },
];

export function UseCases() {
  const track = useRef<HTMLDivElement>(null);

  const scrollBy = (dir: 1 | -1) => {
    track.current?.scrollBy({ left: dir * 320, behavior: "smooth" });
  };

  return (
    <section id="use-cases" className="scroll-mt-20">
      <div className="h-24 md:h-32 lg:h-40" />
      <h2 className="font-headings text-gray-black-soft text-center text-[1.375rem] leading-tight font-semibold md:text-[34px]">
        Where the extra minutes matter
      </h2>
      <div className="h-10 md:h-14" />
      {/* the track sits inside the page container, so the cards line up with
          every other section on both edges */}
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="relative">
          <div
            ref={track}
            className="flex snap-x snap-mandatory gap-6 overflow-x-auto pb-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
          >
            {CARDS.map((c) => (
              <article
                key={c.title}
              className={`bg-gray-black relative flex aspect-[24/31] w-[300px] shrink-0 snap-start flex-col justify-end overflow-hidden rounded-2xl bg-linear-to-b p-6 ${c.tint}`}
            >
              <CardArt
                scene={c.scene}
                className="pointer-events-none absolute -top-2 right-[-18%] w-[86%] text-white/[0.09]"
              />
              <span className="absolute top-6 left-6 flex size-11 items-center justify-center rounded-full border border-white/25 text-white">
                <c.Icon className="size-5" strokeWidth={1.6} />
              </span>
              <h3 className="font-headings relative text-xl leading-snug font-semibold text-white">
                {c.title}
              </h3>
              <p className="text-gray-light relative mt-2.5 text-base leading-relaxed">
                {c.body}
              </p>
            </article>
          ))}
          </div>
          <button
            aria-label="Previous use case"
            onClick={() => scrollBy(-1)}
            className="from-gray-black to-gray-black/70 text-gray-light absolute top-1/2 -left-5 hidden size-11 -translate-y-1/2 items-center justify-center rounded-full bg-linear-to-br shadow-lg transition-transform hover:scale-105 xl:flex"
          >
            <ChevronLeft className="size-5" />
          </button>
          <button
            aria-label="Next use case"
            onClick={() => scrollBy(1)}
            className="from-gray-black to-gray-black/70 text-gray-light absolute top-1/2 -right-5 hidden size-11 -translate-y-1/2 items-center justify-center rounded-full bg-linear-to-br shadow-lg transition-transform hover:scale-105 xl:flex"
          >
            <ChevronRight className="size-5" />
          </button>
        </div>
      </div>
      <div className="h-24 md:h-32 lg:h-40" />
    </section>
  );
}
