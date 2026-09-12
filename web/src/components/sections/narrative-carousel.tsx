"use client";

import { useEffect, useState } from "react";
import { SlideArt, type SlideScene } from "@/components/art/slide-art";

interface Slide {
  heading: string;
  lead: string;
  body: string;
  scene: SlideScene;
  bg: string;
}

const SLIDES: Slide[] = [
  {
    heading: "Attacks are trajectories.",
    lead: "A compromise takes minutes, not milliseconds",
    body: "Recon, then a foothold, then movement sideways. Each step changes how the machine behaves, and it all happens before anything is taken.",
    scene: "trajectory",
    bg: "bg-linear-to-br from-gray-light via-white to-node-blue/50",
  },
  {
    heading: "Detection looks backward.",
    lead: "A signature fires after the packet arrives",
    body: "By the time it does, the trajectory is already running. You are reading history and calling it an alert.",
    scene: "backward",
    bg: "bg-linear-to-br from-gray-light via-white to-gray-medium/40",
  },
  {
    heading: "NIDRA learns the dynamics.",
    lead: "No attack labels, no signature list",
    body: "It learns how a host's state moves from one 30-second window to the next. The training data is ordinary traffic, so there is nothing to keep up to date.",
    scene: "dynamics",
    bg: "bg-linear-to-br from-node-lavender/50 via-white to-primary-pink/40",
  },
  {
    heading: "So it can run the future forward.",
    lead: "About 1000 futures per host, every 30 seconds",
    body: "When enough of them head the same way, you get a risk curve, a band around it, and the number of seconds you have left.",
    scene: "forward",
    bg: "bg-linear-to-br from-node-mint/50 via-white to-node-blue/50",
  },
];

const INTERVAL_MS = 6000;

export function NarrativeCarousel() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const id = setInterval(
      () => setActive((a) => (a + 1) % SLIDES.length),
      INTERVAL_MS,
    );
    return () => clearInterval(id);
  }, []);

  return (
    <section className="lg:grid lg:items-center">
      <div className="h-10 md:h-14 lg:h-20" />
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="relative w-full">
          <div className="relative aspect-[4/5] size-full overflow-hidden rounded-4xl bg-white sm:aspect-[16/11] lg:aspect-[1310/717]">
            {SLIDES.map((slide, i) => (
              <div
                key={slide.heading}
                aria-hidden={i !== active}
                className={`absolute inset-0 transition-opacity duration-700 ${slide.bg} ${
                  i === active ? "opacity-100" : "opacity-0"
                }`}
              >
                <div className="grid h-full grid-cols-1 items-center gap-8 px-8 py-12 sm:px-12 lg:grid-cols-[minmax(0,50%)_1fr] lg:px-20 lg:py-16">
                  <div>
                    <h3 className="font-display text-navy text-[2rem] leading-[1.1] font-semibold italic md:text-[46px] lg:text-[56px]">
                      {slide.heading}
                    </h3>
                    <p className="mt-6 inline-block">
                      <span className="bg-primary-pink/45 text-primary-blue box-decoration-clone rounded-sm px-1.5 py-0.5 text-lg font-semibold md:text-xl">
                        {slide.lead}
                      </span>
                    </p>
                    <p className="text-gray-black mt-4 max-w-xl text-lg leading-relaxed md:text-xl">
                      {slide.body}
                    </p>
                  </div>
                  <div className="hidden items-center justify-center lg:flex">
                    <SlideArt scene={slide.scene} className="w-full max-w-[620px]" />
                  </div>
                </div>
              </div>
            ))}
            <ul className="absolute bottom-5 left-1/2 z-10 flex -translate-x-1/2 items-center justify-center gap-3">
              {SLIDES.map((slide, i) => (
                <li key={slide.heading}>
                  <button
                    type="button"
                    aria-label={`Go to slide ${i + 1}`}
                    onClick={() => setActive(i)}
                    className={`relative block h-2.5 overflow-hidden rounded-full transition-[width] duration-300 ${
                      i === active
                        ? "bg-primary-blue/60 w-7"
                        : "bg-primary-pink w-2.5"
                    }`}
                  />
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
      <div className="h-10 md:h-14 lg:h-20" />
    </section>
  );
}
