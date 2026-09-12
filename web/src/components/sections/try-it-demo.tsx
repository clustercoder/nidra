import { Sparkle, Upload, UploadCloud } from "lucide-react";
import { ForecastConeSvg } from "@/components/forecast-cone-svg";

export function TryItDemo() {
  return (
    <section>
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="h-24 md:h-32 lg:h-40" />
        <h2 className="font-headings text-gray-black-soft text-center text-[1.375rem] leading-tight font-semibold md:text-[34px]">
          PCAP or CSV in, forecast out
        </h2>
        <p className="text-gray-dark mx-auto mt-5 max-w-[820px] text-center text-xl">
          Runs offline. No API key, no cloud round trip.
        </p>
        <div className="h-10 md:h-14" />
        {/* demo shell */}
        <div className="border-gray-medium overflow-hidden rounded-lg border bg-white shadow-sm">
          {/* top bar */}
          <div className="bg-primary-blue flex h-9 items-center px-4">
            <Sparkle className="size-4 fill-white text-white" />
          </div>
          <div className="grid min-h-[540px] grid-cols-1 lg:grid-cols-[305px_1fr]">
            {/* upload sidebar */}
            <div className="border-gray-medium/60 flex flex-col border-b lg:border-r lg:border-b-0">
              <div className="border-gray-medium/60 flex items-center gap-2 border-b px-4 py-4">
                <span className="border-primary-pink flex size-7 items-center justify-center rounded-full border">
                  <Upload className="text-primary-blue size-3.5" />
                </span>
                <span className="text-gray-black-soft text-lg">
                  Run a forecast
                </span>
              </div>
              <div className="flex-1 p-4">
                <div className="border-gray-medium flex h-full min-h-[260px] flex-col items-center justify-center rounded-md border border-dashed px-4 text-center">
                  <UploadCloud
                    className="text-primary-blue size-10"
                    strokeWidth={1.25}
                  />
                  <p className="text-gray-black-soft mt-4 text-base font-medium">
                    Drop a PCAP or CSV of flow records
                  </p>
                  <p className="text-gray-dark mt-2 text-sm">
                    CIC-IDS2017 slices work out of the box
                  </p>
                </div>
              </div>
              <div className="p-4">
                <div
                  className="border-gray-medium text-gray-dark flex items-center justify-between rounded border px-3 py-2 text-base"
                  aria-disabled
                >
                  Replay: Wednesday (60×)
                  <span className="text-gray-medium text-xs">▾</span>
                </div>
              </div>
            </div>
            {/* forecast panel */}
            <div className="p-6">
              <div className="border-gray-medium/70 flex h-full min-h-[480px] flex-col rounded-md border">
                <div className="border-gray-medium/60 text-gray-black-soft border-b px-5 py-4 text-base font-semibold">
                  Infiltration probability timeline
                </div>
                <div className="flex flex-1 items-center justify-center">
                  <ForecastConeSvg className="h-full w-full p-6" />
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* serving callout */}
        <div className="border-gray-light mt-8 flex flex-col gap-4 rounded-xl border bg-white p-6 md:flex-row md:items-start md:gap-8 md:p-8">
          <span className="bg-utility-success-light text-positive inline-flex h-fit shrink-0 items-center rounded-md px-3 py-1 text-sm font-semibold tracking-wide uppercase">
            Serving
          </span>
          <div>
            <p className="text-gray-black-soft text-lg font-semibold">
              NIDRA Predictor
            </p>
            <p className="text-gray-dark mt-2 max-w-3xl text-base leading-relaxed">
              One interface with three methods: forecast, counterfactual,
              explain. It loads once and scores in under 300 ms on CPU, then
              hands back a risk curve, a predicted stage, and the flows behind
              them. Counterfactuals here are model-internal what-ifs.
            </p>
          </div>
        </div>
        <div className="h-24 md:h-32 lg:h-40" />
      </div>
    </section>
  );
}
