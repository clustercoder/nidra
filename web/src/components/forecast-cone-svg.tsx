/**
 * NIDRA's signature visual, hand-drawn as a static SVG.
 *
 * Observed risk (solid) runs up to the "now" rule; from that exact point — the
 * anchor is mandatory, a detached cone is a known failure mode — the projected
 * distribution opens rightward across the K=6 horizon. The projected mean
 * crosses the 0.75 threshold inside the horizon, which is what buys lead time.
 *
 * Y axis is fixed to [0, 1]. Never auto-scale: it makes hosts incomparable.
 *
 * Every number on the face is derived from the series below — the crossing
 * point and the lead-time badge are computed, not typed, so the annotation
 * cannot drift away from the curve it annotates.
 */

const W = 640;
const H = 400;

// plot box
const L = 46; // left gutter for y labels
const R = 604;
const T = 28;
const B = 300; // x-axis; stage strip lives below

const WINDOW_SECONDS = 30;
const RISK_THRESHOLD = 0.75;

/** Observed risk, one value per 30 s window, rising gently toward now. */
const OBSERVED = [0.07, 0.09, 0.08, 0.12, 0.11, 0.17, 0.25, 0.34];
/** Projected mean over the K=6 horizon, and the band half-width at each step. */
const MEAN = [0.4, 0.56, 0.76, 0.79, 0.81, 0.82];
const SPREAD = [0.04, 0.09, 0.14, 0.18, 0.22, 0.26];

const NOW_INDEX = OBSERVED.length - 1;
const STEPS = NOW_INDEX + MEAN.length; // 7 observed steps + 6 projected = 13

/** risk 0..1 → svg y */
const y = (risk: number) => B - risk * (B - T);
/** window index 0..STEPS → svg x, uniform spacing (windows are uniform in time) */
const x = (i: number) => L + (i / STEPS) * (R - L);

/** ~54% of the plot width — the cone gets the rest. */
const NOW_X = x(NOW_INDEX);
const ANCHOR = OBSERVED[NOW_INDEX];

const clamp = (v: number) => Math.max(0, Math.min(1, v));

const observedPath = OBSERVED.map(
  (v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`,
).join(" ");

const meanPath = [
  `M${NOW_X.toFixed(1)} ${y(ANCHOR).toFixed(1)}`,
  ...MEAN.map((v, i) => `L${x(NOW_INDEX + 1 + i).toFixed(1)} ${y(v).toFixed(1)}`),
].join(" ");

const conePath = [
  `M${NOW_X.toFixed(1)} ${y(ANCHOR).toFixed(1)}`,
  ...MEAN.map(
    (v, i) =>
      `L${x(NOW_INDEX + 1 + i).toFixed(1)} ${y(clamp(v + SPREAD[i])).toFixed(1)}`,
  ),
  ...MEAN.map((_, i) => {
    const k = MEAN.length - 1 - i;
    return `L${x(NOW_INDEX + 1 + k).toFixed(1)} ${y(clamp(MEAN[k] - SPREAD[k])).toFixed(1)}`;
  }),
  "Z",
].join(" ");

/**
 * Where the projected mean actually meets the threshold, interpolated between
 * the two straddling windows. Lead time is reported in whole windows, because
 * that is the resolution the model forecasts at.
 */
const crossStep = MEAN.findIndex((v) => v >= RISK_THRESHOLD);
const prevRisk = crossStep === 0 ? ANCHOR : MEAN[crossStep - 1];
const prevX = x(NOW_INDEX + crossStep);
const t = (RISK_THRESHOLD - prevRisk) / (MEAN[crossStep] - prevRisk);
const CROSS_X = prevX + t * (x(NOW_INDEX + 1 + crossStep) - prevX);
const CROSS_Y = y(RISK_THRESHOLD);
const LEAD_WINDOWS = crossStep + 1;
const LEAD_SECONDS = LEAD_WINDOWS * WINDOW_SECONDS;

const BADGE_W = 132;
const BADGE_H = 30;
const BADGE_X = CROSS_X - BADGE_W / 2;
const BADGE_Y = CROSS_Y - 56;

const STAGES: { label: string; tone: "benign" | "early" | "late" }[] = [
  { label: "benign", tone: "benign" },
  { label: "recon", tone: "benign" },
  { label: "initial_access", tone: "early" },
  { label: "lateral", tone: "early" },
  { label: "c2", tone: "late" },
  { label: "exfil", tone: "late" },
];

const STAGE_FILL: Record<string, string> = {
  benign: "fill-positive/25",
  early: "fill-threshold/25",
  late: "fill-negative/25",
};

export function ForecastConeSvg({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={className}
      role="img"
      aria-label={`Forecast chart: observed risk rises to the now line, then a widening band of simulated trajectories projects forward three minutes and crosses the ${RISK_THRESHOLD} risk threshold ${LEAD_WINDOWS} windows out, giving about ${LEAD_SECONDS} seconds of lead time.`}
    >
      {/* grid */}
      <g className="stroke-gray-light" strokeWidth="1">
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <line key={v} x1={L} x2={R} y1={y(v)} y2={y(v)} />
        ))}
      </g>
      <g className="fill-gray-dark" fontSize="11">
        {[0, 0.5, 1].map((v) => (
          <text key={v} x={L - 10} y={y(v) + 4} textAnchor="end">
            {v.toFixed(1)}
          </text>
        ))}
        <text x={L - 34} y={T - 12} fontSize="10.5">
          risk
        </text>
      </g>

      {/* axis */}
      <line x1={L} x2={R} y1={B} y2={B} className="stroke-gray-medium" strokeWidth="1" />

      {/* uncertainty cone — anchored at the observed value on the now rule */}
      <path d={conePath} className="fill-projected/20" />

      {/* projected mean */}
      <path
        d={meanPath}
        fill="none"
        className="stroke-projected"
        strokeWidth="2.5"
        strokeDasharray="7 5"
        strokeLinecap="round"
      />

      {/* observed risk */}
      <path
        d={observedPath}
        fill="none"
        className="stroke-observed"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={NOW_X} cy={y(ANCHOR)} r="4" className="fill-observed" />

      {/* threshold */}
      <line
        x1={L}
        x2={R}
        y1={CROSS_Y}
        y2={CROSS_Y}
        className="stroke-threshold"
        strokeWidth="1.5"
        strokeDasharray="4 4"
      />
      <text x={L + 6} y={CROSS_Y - 7} className="fill-threshold" fontSize="11">
        risk threshold {RISK_THRESHOLD}
      </text>

      {/* now rule */}
      <line
        x1={NOW_X}
        x2={NOW_X}
        y1={T - 10}
        y2={B}
        className="stroke-gray-dark"
        strokeWidth="1"
        strokeDasharray="3 4"
      />
      <text x={NOW_X + 6} y={T - 2} className="fill-gray-dark" fontSize="11">
        now
      </text>
      <text x={NOW_X - 6} y={T - 2} textAnchor="end" className="fill-gray-dark" fontSize="11">
        observed
      </text>

      {/* crossing marker + lead-time badge */}
      <circle cx={CROSS_X} cy={CROSS_Y} r="5" className="fill-threshold" />
      <circle
        cx={CROSS_X}
        cy={CROSS_Y}
        r="10"
        fill="none"
        className="stroke-threshold/50"
        strokeWidth="1.5"
      />
      <line
        x1={CROSS_X}
        x2={CROSS_X}
        y1={BADGE_Y + BADGE_H}
        y2={CROSS_Y - 11}
        className="stroke-threshold/60"
        strokeWidth="1.2"
      />
      <rect
        x={BADGE_X}
        y={BADGE_Y}
        width={BADGE_W}
        height={BADGE_H}
        rx={BADGE_H / 2}
        fill="white"
        className="stroke-threshold"
        strokeWidth="1.2"
      />
      <text
        x={CROSS_X}
        y={BADGE_Y + 19}
        textAnchor="middle"
        className="fill-threshold"
        fontSize="12.5"
      >
        lead time ~{LEAD_SECONDS} s
      </text>

      {/* horizon bracket */}
      <g className="fill-gray-dark" fontSize="11">
        <line
          x1={NOW_X}
          x2={R}
          y1={B + 16}
          y2={B + 16}
          className="stroke-gray-medium"
          strokeWidth="1"
        />
        <text x={(NOW_X + R) / 2} y={B + 12} textAnchor="middle">
          K = {MEAN.length} windows · 3-minute horizon
        </text>
      </g>

      {/* stage strip — predicted attack-stage lifecycle */}
      <g transform={`translate(${L} ${B + 34})`}>
        {STAGES.map((stage, i) => {
          const segW = (R - L) / STAGES.length - 6;
          return (
            <g key={stage.label} transform={`translate(${i * ((R - L) / STAGES.length)} 0)`}>
              <rect
                width={segW}
                height="20"
                rx="5"
                className={`${STAGE_FILL[stage.tone]} stroke-gray-light`}
                strokeWidth="1"
              />
              <text
                x={segW / 2}
                y="14"
                textAnchor="middle"
                className="fill-gray-black-soft"
                fontSize="10"
              >
                {stage.label}
              </text>
            </g>
          );
        })}
        <text x="0" y="38" className="fill-gray-dark" fontSize="10.5">
          predicted attack stage · 6-stage lifecycle
        </text>
      </g>
    </svg>
  );
}
