import { Activity, ArrowRight } from "lucide-react";

export function CtaBanner() {
  return (
    <section className="bg-primary-blue relative">
      {/* right-half gradient wash */}
      <div className="from-primary-blue to-primary-pink/70 absolute top-0 right-0 h-full w-1/2 bg-linear-to-br from-45%" />
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="flex flex-col items-center justify-center py-8 lg:flex-row lg:gap-20 lg:py-0">
          <div className="relative flex w-[103px] shrink-0 items-center justify-center">
            <span className="flex size-20 items-center justify-center rounded-full bg-white/20 text-white">
              <Activity className="size-10" strokeWidth={1.5} aria-hidden />
            </span>
          </div>
          <div className="relative text-center text-white lg:py-8 lg:text-left">
            <p className="text-primary-pink text-lg">
              See it run
            </p>
            <h2 className="font-headings text-[1.375rem]/[1.85rem] font-semibold text-white md:text-[34px] md:leading-tight">
              Replay a real attack at 60×
            </h2>
          </div>
          <div className="relative mt-4 lg:mt-0">
            <a
              href="/demo"
              className="group/button bg-primary-pink text-navy relative inline-flex items-center justify-start gap-1.5 rounded-full border border-transparent px-[22px] py-3 font-medium transition-all duration-200 hover:scale-[1.03]"
            >
              Open the demo
              <ArrowRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
