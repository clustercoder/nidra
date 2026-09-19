/**
 * Score an uploaded capture with the real model.
 *
 * This route is a thin adapter over `scripts/analyze_pcap.py` and deliberately
 * nothing more: the analysis, the flow reconstruction and the predictor all
 * live in Python next to the training code, so there is no second
 * implementation of the pipeline that could drift from the one the numbers on
 * /demo came from. The route's whole job is to accept a file safely, hand its
 * path to that script, and pass the script's JSON back.
 *
 * It runs the capture locally, which is what makes `npm run dev` enough to try
 * it. The uploaded file never leaves the machine and is deleted as soon as the
 * analysis returns, including on failure.
 *
 * What is checked before anything is spawned:
 *  - the request is multipart with exactly one file
 *  - the file is under MAX_UPLOAD_BYTES
 *  - the first bytes are a real pcap/pcapng magic, not just a `.pcap` name
 *  - the path handed to the subprocess is one this route generated
 *
 * The client's filename is used for display only. It never names anything on
 * disk and never reaches the command line — the argv is a fixed array, so
 * there is no shell for a crafted name to reach.
 */

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import { NextResponse } from "next/server";

/** Needs a real filesystem and a subprocess; not an edge function. */
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Big enough for a few minutes of real traffic, small enough to stay a demo. */
const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;

/** libpcap (both byte orders), nanosecond libpcap, and pcapng. */
const PCAP_MAGICS = [
  "d4c3b2a1",
  "a1b2c3d4",
  "4d3cb2a1",
  "a1b23c4d",
  "0a0d0d0a",
];

/**
 * tshark on a large capture is the slow step, and the model then scores every
 * window of every host. A capture near the size cap can legitimately take
 * minutes; past this it is a hang, not slow progress.
 */
const ANALYSIS_TIMEOUT_MS = 10 * 60 * 1000;

/** Guards against a runaway subprocess filling memory through the pipe. */
const MAX_STDOUT_BYTES = 64 * 1024 * 1024;

type Failure = { status: number; error: string };

function repoRoot(): string {
  // `next dev` runs with cwd=web/. An explicit override wins so the route also
  // works when the app is started from somewhere else.
  return process.env.NIDRA_REPO_ROOT ?? path.resolve(process.cwd(), "..");
}

function pythonBin(): string {
  return process.env.NIDRA_PYTHON ?? path.join(repoRoot(), ".venv", "bin", "python");
}

function looksLikePcap(head: Uint8Array): boolean {
  const hex = Array.from(head.slice(0, 4))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return PCAP_MAGICS.includes(hex);
}

/** Run the analyser and return its stdout, or a failure with a shown reason. */
async function runAnalyser(pcapPath: string): Promise<{ stdout: string } | Failure> {
  return new Promise((resolve) => {
    const child = spawn(
      pythonBin(),
      ["scripts/analyze_pcap.py", pcapPath],
      { cwd: repoRoot(), stdio: ["ignore", "pipe", "pipe"] },
    );

    let stdout = "";
    let stderr = "";
    let overflowed = false;
    const timer = setTimeout(() => child.kill("SIGKILL"), ANALYSIS_TIMEOUT_MS);

    child.stdout.on("data", (chunk: Buffer) => {
      if (stdout.length + chunk.length > MAX_STDOUT_BYTES) {
        overflowed = true;
        child.kill("SIGKILL");
        return;
      }
      stdout += chunk.toString();
    });
    // Kept for the server log on a crash; never returned to the client, which
    // would leak absolute paths and library internals.
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = (stderr + chunk.toString()).slice(-8192);
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      console.error("analyze-pcap: could not start the analyser", err);
      resolve({
        status: 500,
        error:
          `Could not start the analyser (${pythonBin()}). The Python environment has to be " +
          "set up for capture analysis — see the README's local setup.`,
      });
    });

    child.on("close", (code, signal) => {
      clearTimeout(timer);
      if (overflowed) {
        resolve({ status: 413, error: "The analysis produced more output than this route accepts." });
        return;
      }
      if (signal === "SIGKILL") {
        resolve({ status: 504, error: "The analysis took too long and was stopped." });
        return;
      }
      // Exit 1 is the analyser's own refusal: its stdout carries {"error": ...}
      // written for the person who uploaded the file.
      if (code !== 0 && !stdout.trim().startsWith("{")) {
        console.error("analyze-pcap: analyser exited %s\n%s", code, stderr);
        resolve({ status: 500, error: "The analysis failed. The server log has the details." });
        return;
      }
      resolve({ stdout });
    });
  });
}

export async function POST(request: Request) {
  const declared = Number(request.headers.get("content-length") ?? 0);
  if (declared > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: `Captures are limited to ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.` },
      { status: 413 },
    );
  }

  let file: File;
  try {
    const form = await request.formData();
    const entry = form.get("capture");
    if (!(entry instanceof File)) {
      return NextResponse.json({ error: "Send the capture as the `capture` field." }, { status: 400 });
    }
    file = entry;
  } catch {
    return NextResponse.json({ error: "Could not read the upload." }, { status: 400 });
  }

  if (file.size === 0) {
    return NextResponse.json({ error: "That file is empty." }, { status: 400 });
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { error: `Captures are limited to ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.` },
      { status: 413 },
    );
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  if (!looksLikePcap(bytes)) {
    return NextResponse.json(
      {
        error:
          "That is not a pcap. The file has to start with a libpcap or pcapng " +
          "magic — a `.pcap` extension on something else will not do.",
      },
      { status: 415 },
    );
  }

  const dir = await mkdtemp(path.join(tmpdir(), "nidra-upload-"));
  // Our name, not the client's: the uploaded filename is display text only.
  const pcapPath = path.join(dir, `${randomUUID()}.pcap`);
  try {
    await writeFile(pcapPath, bytes);
    const result = await runAnalyser(pcapPath);
    if ("error" in result) {
      return NextResponse.json({ error: result.error }, { status: result.status });
    }

    let payload: unknown;
    try {
      payload = JSON.parse(result.stdout);
    } catch {
      console.error("analyze-pcap: analyser did not return JSON");
      return NextResponse.json({ error: "The analysis returned nothing usable." }, { status: 500 });
    }
    if (payload && typeof payload === "object" && "error" in payload) {
      // The analyser's own refusal, written for the uploader to act on.
      return NextResponse.json(payload, { status: 422 });
    }
    // Attach the name the person recognises; the analyser only ever saw ours.
    if (payload && typeof payload === "object") {
      const source = (payload as { source?: Record<string, unknown> }).source;
      const capture = source?.capture as Record<string, unknown> | undefined;
      if (capture) capture.filename = file.name;
    }
    return NextResponse.json(payload);
  } finally {
    await rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}
