"use client";

/**
 * Holds which capture the console is showing: the committed CIC-IDS2017
 * replay, or one the visitor just analysed.
 *
 * The console itself is deliberately unaware of the difference — it takes
 * hosts, forecasts and a model, and both sources supply exactly that. Keeping
 * the switch here is what stops an "is this an upload?" flag from spreading
 * through the rendering code, where it would eventually be read inconsistently
 * and present an upload with the replay's confidence.
 *
 * The console is remounted (via `key`) when the source changes, because its
 * clock, selected host and selected window all index into the data it was
 * given; carrying them across two unrelated captures would point them at the
 * wrong moment or past the end.
 */

import * as React from "react";

import {
  alignedHosts,
  forecastsByHost,
  type ExplainInfo,
  type Forecast,
  type ModelInfo,
  type UploadedReplay,
} from "@/lib/demo-replay";

import { DemoConsole } from "./demo-console";

export function DemoWorkspace({
  model,
  hosts,
  byHost,
  explanations,
  sourceSummary,
  initial,
}: {
  model: ModelInfo;
  hosts: string[];
  byHost: Record<string, Forecast[]>;
  explanations: Record<string, ExplainInfo>;
  sourceSummary: string;
  initial: { host?: string; t: number; paused: boolean };
}) {
  const [uploaded, setUploaded] = React.useState<UploadedReplay | null>(null);
  const [rejected, setRejected] = React.useState<string | null>(null);

  /* Checked when the analysis lands rather than while rendering: a capture the
     console cannot rank is a fact about the upload, and the visitor needs to
     be told which capture was rejected and why. */
  const accept = React.useCallback((replay: UploadedReplay) => {
    const grouped = forecastsByHost(replay.forecasts);
    if (!alignedHosts(replay.hosts, grouped)) {
      setRejected(
        `${replay.source.capture.filename} analysed, but no two hosts in it share a ` +
          "window timeline, so there is nothing the console can rank against anything " +
          "else. Try a capture with more concurrent traffic.",
      );
      return;
    }
    setRejected(null);
    setUploaded(replay);
  }, []);

  const shown = React.useMemo(() => {
    if (!uploaded) {
      return { key: "builtin", model, hosts, byHost, explanations, sourceSummary, initial };
    }
    const grouped = forecastsByHost(uploaded.forecasts);
    // Non-null: `accept` is the only way `uploaded` is set, and it checks this.
    const aligned = alignedHosts(uploaded.hosts, grouped)!;
    return {
      key: `upload:${uploaded.source.capture.filename}:${uploaded.source.capture.packets}`,
      model: uploaded.model,
      hosts: aligned.hosts,
      byHost: grouped,
      explanations: uploaded.explanations,
      sourceSummary: uploaded.source.summary,
      // An analysed capture opens paused on its first window: unlike the
      // rehearsed replay there is no known moment to run up to, and a viewer
      // wants to read the numbers before the clock moves them.
      initial: { t: 0, paused: true },
    };
  }, [uploaded, model, hosts, byHost, explanations, sourceSummary, initial]);

  return (
    <>
      {rejected && (
        <div
          role="status"
          className="border-negative-lit/40 bg-negative-lit/10 text-console-text px-5 py-2 text-xs"
        >
          {rejected}
        </div>
      )}
      <DemoConsole
        key={shown.key}
        model={shown.model}
        hosts={shown.hosts}
        byHost={shown.byHost}
        explanations={shown.explanations}
        sourceSummary={shown.sourceSummary}
        initial={shown.initial}
        uploaded={uploaded}
        onAnalysed={accept}
        onClearUpload={() => {
          setRejected(null);
          setUploaded(null);
        }}
      />
    </>
  );
}
