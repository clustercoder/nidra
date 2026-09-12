import type { Metadata } from "next";

/* This route reads the committed fixture and nothing else — no API client, no
   websocket. /demo must render with every container stopped. */
import { forecastsByHost, replay } from "@/lib/demo-replay";

import { DemoConsole } from "./demo-console";

export const metadata: Metadata = {
  title: "Demo console — NIDRA",
  description:
    "Public replay console. A synthesised CIC-IDS2017-Wednesday-shaped capture driven through the stub pipeline; values are illustrative.",
};

function first(v: string | string[] | undefined) {
  return Array.isArray(v) ? v[0] : v;
}

export default async function DemoPage({ searchParams }: PageProps<"/demo">) {
  const params = await searchParams;
  const byHost = forecastsByHost();

  // Every host must carry the same window sequence, or ranking by index would
  // silently compare different moments in time.
  const hosts = replay.hosts.filter((h) => byHost[h]?.length);
  const lengths = new Set(hosts.map((h) => byHost[h].length));
  if (hosts.length === 0 || lengths.size !== 1) {
    throw new Error("demo-replay.json hosts are missing or misaligned");
  }
  const windowCount = [...lengths][0];

  /* ?host=&t= — a rehearsed demo opens on the interesting window instead of
     scrubbing to it on stage. Invalid values fall back, and an explicit ?t=
     starts paused there. */
  const hostParam = first(params.host);
  const tParam = first(params.t);
  const tParsed = tParam === undefined ? Number.NaN : Number.parseInt(tParam, 10);
  const tValid = Number.isInteger(tParsed) && tParsed >= 0 && tParsed < windowCount;

  return (
    <DemoConsole
      model={replay.model}
      hosts={hosts}
      byHost={byHost}
      explanations={replay.explanations}
      sourceSummary={replay.source.summary}
      initial={{
        host: hostParam && hosts.includes(hostParam) ? hostParam : undefined,
        t: tValid ? tParsed : 0,
        paused: tValid,
      }}
    />
  );
}
