import { ChevronRight } from "lucide-react";
import { ForecastConeSvg } from "@/components/forecast-cone-svg";

export function FinalCta() {
  return (
    <section className="gradient-light-tl overflow-hidden">
      <div className="h-24 md:h-32 lg:h-44" />
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <h2 className="font-headings text-gray-black-soft text-center text-[1.375rem] leading-tight font-semibold md:text-[34px]">
          Ready to see NIDRA in action?
        </h2>
        <div className="h-12 md:h-16 lg:h-24" />
        <div className="relative">
          <div className="mx-auto aspect-video w-full max-w-[500px]">
            <div className="border-primary-blue relative aspect-video w-full overflow-hidden rounded-lg border bg-white md:shadow-lg md:shadow-gray-black/30">
              <ForecastConeSvg className="h-full w-full bg-white p-4" />
            </div>
          </div>
        </div>
        <div className="h-12 md:h-16 lg:h-24" />
        <div className="text-center">
          <a
            href="/demo"
            className="group/button bg-primary-blue relative inline-flex items-center gap-1.5 rounded-full px-[22px] py-3 text-base font-medium text-white transition-all duration-200 hover:scale-[1.03] hover:bg-[rgb(24,64,180)]"
          >
            Run a forecast
            <ChevronRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
          </a>
        </div>
      </div>
      <div className="h-24 md:h-32 lg:h-44" />
    </section>
  );
}
