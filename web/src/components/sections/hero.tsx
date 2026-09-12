import { ArrowRight } from "lucide-react";
import { HeroArt } from "@/components/art/hero-art";

export function Hero() {
  return (
    <section className="isolate overflow-hidden bg-white">
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="pt-10 md:py-20 lg:py-16">
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2 md:gap-10 lg:grid-cols-[580px_1fr] lg:items-center">
            {/* Left column */}
            <div className="relative z-10 space-y-3">
              <h1 className="font-headings text-gray-black-soft text-[2rem] leading-tight font-semibold lg:text-[52px]">
                See the intrusion{" "}
                <span className="text-primary-blue relative whitespace-nowrap">
                  <span className="bg-primary-pink/40 absolute inset-x-[-4px] inset-y-1 -z-10 rounded-sm" />
                  before
                </span>
{" "}
                it happens
              </h1>
              <p className="text-gray-black text-xl leading-[1.6]">
                Every 30 seconds NIDRA turns each host&apos;s traffic into 45
                numbers. It has learned how those numbers move, so it can run
                them forward three minutes and say where the host is heading.
                In the Wednesday replay that was{" "}
                <strong className="font-semibold">
                  90 seconds of warning before the scan became an intrusion
                </strong>
                .
              </p>
              <div className="mt-7! md:max-w-[450px]">
                <div className="flex flex-wrap items-center gap-3">
                  <a
                    href="/demo"
                    className="group/button bg-primary-blue inline-flex h-12 shrink-0 items-center gap-1.5 rounded-full px-7 text-base font-medium text-white transition-all duration-200 hover:scale-[1.03] hover:bg-[rgb(24,64,180)]"
                  >
                    Run a forecast
                    <ArrowRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
                  </a>
                  <a
                    href="#how-it-works"
                    className="border-gray-medium text-gray-black-soft hover:border-primary-blue inline-flex h-12 shrink-0 items-center rounded-full border px-7 text-base font-medium transition-colors"
                  >
                    How it works
                  </a>
                </div>
              </div>
            </div>

            {/* Right column — illustration. The chart lives in the demo shell. */}
            <div className="relative min-h-[230px] md:min-h-[400px] lg:min-h-[450px]">
              <div className="hero-float pointer-events-none absolute inset-0">
                <HeroArt className="h-full w-full" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
