import { SiteHeader } from "@/components/site-header";
import { Hero } from "@/components/sections/hero";
import { StackRow } from "@/components/sections/stack-row";
import { EvidenceTeaser } from "@/components/sections/evidence-teaser";
import { NarrativeCarousel } from "@/components/sections/narrative-carousel";
import { WorldModelStages } from "@/components/sections/world-model-stages";
import { TryItDemo } from "@/components/sections/try-it-demo";
import { CtaBanner } from "@/components/sections/cta-banner";
import { HowNidraWorks } from "@/components/sections/how-nidra-works";
import { UseCases } from "@/components/sections/use-cases";
import { ForecastConsole } from "@/components/sections/forecast-console";
import { Experiments } from "@/components/sections/experiments";
import { FinalCta } from "@/components/sections/final-cta";
import { SiteFooter } from "@/components/sections/site-footer";

export default function Home() {
  return (
    <main>
      <SiteHeader />
      <Hero />
      <StackRow />
      <EvidenceTeaser />
      <NarrativeCarousel />
      <WorldModelStages />
      <TryItDemo />
      <CtaBanner />
      <HowNidraWorks />
      <UseCases />
      <ForecastConsole />
      <Experiments />
      <FinalCta />
      <SiteFooter />
    </main>
  );
}
