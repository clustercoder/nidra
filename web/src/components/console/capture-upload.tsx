"use client";

/**
 * Drop a capture in, score it with the real model, show it in this console.
 *
 * The upload goes to /api/analyze-pcap, which runs `scripts/analyze_pcap.py`
 * locally — the same predictor the committed replay was scored with. Nothing
 * is sent anywhere; the file is deleted as soon as the analysis returns.
 *
 * The panel's other job is to stop an analysed capture from reading like the
 * labelled replay. The result carries measured fidelity notes from
 * `scripts/validate_flow_assembly.py` and they are rendered before the console
 * is offered, not tucked behind a link: an uploaded capture has no ground
 * truth and its flow features are reconstructed, and someone looking at the
 * resulting risk curve cannot tell either of those by looking.
 */

import * as React from "react";
import { AlertTriangle, FileUp, Loader2, ShieldCheck } from "lucide-react";

import type { UploadedReplay } from "@/lib/demo-replay";

const ACCEPT = ".pcap,.pcapng,.cap";

type State =
  | { phase: "idle" }
  | { phase: "running"; filename: string }
  | { phase: "failed"; message: string };

export function CaptureUpload({
  active,
  onAnalysed,
  onClear,
}: {
  /** The capture currently being shown, if the console is showing one. */
  active: UploadedReplay | null;
  onAnalysed: (replay: UploadedReplay) => void;
  onClear: () => void;
}) {
  const [state, setState] = React.useState<State>({ phase: "idle" });
  const [dragging, setDragging] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const analyse = React.useCallback(
    async (file: File) => {
      setState({ phase: "running", filename: file.name });
      const body = new FormData();
      body.append("capture", file);
      try {
        const res = await fetch("/api/analyze-pcap", { method: "POST", body });
        const payload = await res.json().catch(() => null);
        if (!res.ok) {
          setState({
            phase: "failed",
            message:
              (payload as { error?: string } | null)?.error ??
              `The analysis failed (HTTP ${res.status}).`,
          });
          return;
        }
        setState({ phase: "idle" });
        onAnalysed(payload as UploadedReplay);
      } catch {
        setState({
          phase: "failed",
          message: "Could not reach the analyser. Is the dev server still running?",
        });
      }
    },
    [onAnalysed],
  );

  const running = state.phase === "running";

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!running) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files?.[0];
          if (file && !running) void analyse(file);
        }}
        className={`border-console-line bg-console-surface rounded-lg border border-dashed p-8 text-center transition-colors ${
          dragging ? "border-observed-lit bg-console-raised" : ""
        }`}
      >
        {running ? (
          <div className="text-console-muted flex flex-col items-center gap-3 text-sm">
            <Loader2 className="text-observed-lit size-6 animate-spin" />
            <p className="text-console-text">Analysing {state.filename}</p>
            <p className="text-xs">
              tshark reads every packet, the flows are rebuilt from them, and the
              5-seed ensemble scores every window of every host. A few minutes
              on a large capture.
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <FileUp className="text-console-muted size-6" />
            <p className="text-console-text text-sm">
              Drop a .pcap or .pcapng here, or{" "}
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="text-observed-lit underline underline-offset-2"
              >
                choose a file
              </button>
            </p>
            <p className="text-console-muted text-xs">
              Runs locally through the same model this console is already
              showing. The file is not uploaded anywhere and is deleted when the
              analysis finishes. Up to 200 MB; needs at least 15 minutes of
              traffic from one host, which is the model&apos;s context window.
            </p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            // Reset so choosing the same file twice fires again.
            e.target.value = "";
            if (file) void analyse(file);
          }}
        />
      </div>

      {state.phase === "failed" && (
        <div className="border-negative-lit/40 bg-negative-lit/5 flex gap-3 rounded-lg border p-4">
          <AlertTriangle className="text-negative-lit mt-0.5 size-4 shrink-0" />
          <p className="text-console-text text-xs leading-relaxed">{state.message}</p>
        </div>
      )}

      {active && (
        <div className="border-console-line bg-console-surface space-y-3 rounded-lg border p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-console-text flex items-center gap-2 text-sm">
              <ShieldCheck className="text-positive-lit size-4" />
              Showing {active.source.capture.filename}
            </p>
            <button
              type="button"
              onClick={onClear}
              className="border-console-line text-console-muted hover:text-console-text rounded-md border px-3 py-1 text-xs"
            >
              Back to the CIC-IDS2017 replay
            </button>
          </div>
          <dl className="text-console-muted grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-4">
            <Fact label="Packets" value={active.source.capture.packets.toLocaleString()} />
            <Fact label="Hosts in capture" value={String(active.source.capture.hosts_in_capture)} />
            <Fact label="Hosts shown" value={String(active.source.capture.hosts_shown)} />
            <Fact label="Windows" value={String(active.source.capture.window_count)} />
          </dl>
          <div>
            <p className="text-console-text text-xs font-medium">
              What this page cannot tell you
            </p>
            <ul className="text-console-muted mt-1 list-disc space-y-1 pl-4 text-[11px] leading-relaxed">
              {active.source.fidelity.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] uppercase tracking-wide">{label}</dt>
      <dd className="text-console-text font-mono tabular-nums">{value}</dd>
    </div>
  );
}
