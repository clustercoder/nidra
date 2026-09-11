"use client";

import type {
  Forecast,
  ModelConfig,
  ObservedPoint,
} from "@/lib/demo-replay";
import { clockOf } from "@/lib/demo-replay";

/**
 * The console's forecast chart, drawn from real fixture values.
 *
 * Solid observed history runs to the now rule; the confidence band opens from
 * the observed value at that exact point (a detached cone is a known failure
 * mode) and the dashed mean runs through it. Y is pinned to [0, 1] — never
 * auto-scaled, because that makes hosts incomparable and flat curves look
 * dramatic.
 */

const W = 900;
const H = 320;
const L = 52;
const R = W - 18;
const T = 22;
const B = 250;

const y = (risk: number) => B - Math.max(0, Math.min(1, risk)) * (B - T);

export function ForecastChart({
  forecast,
  geometry,
  observed,
  reality,
  showReality,
}: {
  forecast: Forecast;
  geometry: ModelConfig;
  observed: ObservedPoint[];
  reality?: ObservedPoint[];
  showReality: boolean;
}) {
  const K = forecast.horizons.length;
  /* The axis holds a fixed L + K slots, so the geometry does not lurch about as
     the replay fills up. History grows leftward from the now rule; the horizon
     always occupies the same slice on the right. */
  const CTX = geometry.context_L;
  const total = CTX + K;
  const x = (i: number) => L + (i / total) * (R - L);
  const NOW_I = CTX;
  const NOW_X = x(NOW_I);
  // draw at most the last CTX observed windows, ending at the now rule
  const shown = observed.slice(Math.max(0, observed.length - CTX - 1));
  const firstI = NOW_I - (shown.length - 1);

  const anchor = forecast.observed_risk;
  const hz = forecast.horizons;

  const obsPath = shown
    .map(
      (p, i) =>
        `${i === 0 ? "M" : "L"}${x(firstI + i).toFixed(1)} ${y(p.risk).toFixed(1)}`,
    )
    .join(" ");

  const meanPath = [
    `M${NOW_X.toFixed(1)} ${y(anchor).toFixed(1)}`,
    ...hz.map((h, i) => `L${x(NOW_I + 1 + i).toFixed(1)} ${y(h.p_compromise).toFixed(1)}`),
  ].join(" ");

  const bandPath = [
    `M${NOW_X.toFixed(1)} ${y(anchor).toFixed(1)}`,
    ...hz.map((h, i) => `L${x(NOW_I + 1 + i).toFixed(1)} ${y(h.ci_high).toFixed(1)}`),
    ...hz
      .slice()
      .reverse()
      .map((h, i) => `L${x(NOW_I + K - i).toFixed(1)} ${y(h.ci_low).toFixed(1)}`),
    "Z",
  ].join(" ");

  const thr = geometry.risk_threshold;
  const lead = forecast.lead_time_s;

  /* The marker is placed from lead_time_s, not by interpolating the mean.
     lead_time_s is the backend's answer under the m-consecutive-windows rule,
     and when risk is already above threshold at the origin an interpolated
     crossing lands behind the now rule — a crossing in the past, which is not
     a thing. Deriving from the reported number keeps badge and marker agreeing. */
  const cross =
    lead === null
      ? null
      : { cx: x(NOW_I + lead / geometry.window_delta), cy: y(thr) };

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={`Forecast for ${forecast.host_id} at ${clockOf(forecast.origin_ts)} UTC. Observed risk ${forecast.observed_risk.toFixed(3)}. ${
          lead === null
            ? `The projected mean stays below the ${thr} threshold across the ${K}-window horizon.`
            : `The projected mean crosses ${thr} with ${lead} seconds of lead time.`
        }`}
      >
        {/* grid + y labels */}
        <g className="stroke-console-line" strokeWidth="1">
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <line key={v} x1={L} x2={R} y1={y(v)} y2={y(v)} />
          ))}
        </g>
        <g className="fill-console-muted" fontSize="11">
          {[0, 0.5, 1].map((v) => (
            <text key={v} x={L - 10} y={y(v) + 4} textAnchor="end">
              {v.toFixed(2)}
            </text>
          ))}
          <text x={L - 40} y={T - 8} fontSize="10.5">
            risk
          </text>
        </g>
        <line x1={L} x2={R} y1={B} y2={B} className="stroke-console-line" strokeWidth="1" />

        {/* confidence band, anchored on the observed value at now */}
        <path d={bandPath} className="fill-projected-lit/20" />

        {/* projected mean */}
        <path
          d={meanPath}
          fill="none"
          className="stroke-projected-lit"
          strokeWidth="2.5"
          strokeDasharray="7 5"
          strokeLinecap="round"
        />

        {/* observed history */}
        <path
          d={obsPath}
          fill="none"
          className="stroke-observed-lit"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx={NOW_X} cy={y(anchor)} r="4.5" className="fill-observed-lit" />

        {/* threshold */}
        <line
          x1={L}
          x2={R}
          y1={y(thr)}
          y2={y(thr)}
          className="stroke-threshold-lit"
          strokeWidth="1.5"
          strokeDasharray="4 4"
        />
        <text x={L + 6} y={y(thr) - 7} className="fill-threshold-lit" fontSize="11">
          risk threshold {thr}
        </text>

        {/* now rule */}
        <line
          x1={NOW_X}
          x2={NOW_X}
          y1={T - 8}
          y2={B}
          className="stroke-console-muted"
          strokeWidth="1"
          strokeDasharray="3 4"
        />
        <text x={NOW_X + 6} y={T} className="fill-console-muted" fontSize="11">
          now
        </text>

        {/* crossing + lead time */}
        {cross && lead !== null && (
          <g>
            <circle cx={cross.cx} cy={cross.cy} r="5" className="fill-threshold-lit" />
            <circle
              cx={cross.cx}
              cy={cross.cy}
              r="10"
              fill="none"
              className="stroke-threshold-lit/50"
              strokeWidth="1.5"
            />
            <g transform={`translate(${Math.min(cross.cx, R - 74) - 66} ${cross.cy - 54})`}>
              <rect
                width="132"
                height="28"
                rx="14"
                className="fill-console-raised stroke-threshold-lit"
                strokeWidth="1.2"
              />
              <text x="66" y="18" textAnchor="middle" className="fill-threshold-lit" fontSize="12">
                lead time {lead} s
              </text>
            </g>
          </g>
        )}

        {/* what actually happened — drawn where it fell, inside the band or not */}
        {showReality &&
          reality?.map((p, i) => {
            const cx = x(NOW_I + 1 + i);
            const cy = y(p.risk);
            const h = hz[i];
            const inside = h && p.risk >= h.ci_low && p.risk <= h.ci_high;
            return (
              <g key={p.ts} className={inside ? "stroke-positive" : "stroke-negative"}>
                <line x1={cx - 5} y1={cy - 5} x2={cx + 5} y2={cy + 5} strokeWidth="2.5" strokeLinecap="round" />
                <line x1={cx + 5} y1={cy - 5} x2={cx - 5} y2={cy + 5} strokeWidth="2.5" strokeLinecap="round" />
              </g>
            );
          })}

        {/* horizon bracket */}
        <g className="fill-console-muted" fontSize="11">
          <line x1={NOW_X} x2={R} y1={B + 15} y2={B + 15} className="stroke-console-line" strokeWidth="1" />
          <text x={(NOW_X + R) / 2} y={B + 32} textAnchor="middle">
            K = {K} windows · {(K * geometry.window_delta) / 60}-minute horizon
          </text>
        </g>
        <text x={L} y={B + 32} className="fill-console-muted" fontSize="11">
          {observed.length} observed windows
        </text>
      </svg>
    </figure>
  );
}
