/**
 * The traffic-to-forecast line. One continuous wire with instruments on it, and
 * pill labels naming what travels between them. Feature positions line up with
 * the four column headings underneath.
 */

const W = 1200;
const H = 250;
const Y = 148; // the wire

// column centres, matching the 4-up grid below
const C = [150, 450, 750, 1050];

function Pill({ x, label }: { x: number; label: string }) {
  const w = label.length * 7.4 + 26;
  return (
    <g transform={`translate(${x - w / 2} ${Y - 74})`}>
      <rect
        width={w}
        height="26"
        rx="13"
        fill="white"
        className="stroke-primary-blue"
        strokeWidth="1.5"
      />
      <text
        x={w / 2}
        y="17"
        textAnchor="middle"
        className="fill-primary-blue"
        fontSize="11"
        letterSpacing="0.06em"
      >
        {label}
      </text>
      <line
        x1={w / 2}
        y1="26"
        x2={w / 2}
        y2="74"
        className="stroke-gray-medium"
        strokeWidth="1.5"
        strokeDasharray="3 4"
      />
    </g>
  );
}

function Caption({ x, label }: { x: number; label: string }) {
  return (
    <text
      x={x}
      y={Y + 74}
      textAnchor="middle"
      className="fill-primary-blue"
      fontSize="12.5"
      fontWeight="600"
      letterSpacing="0.06em"
    >
      {label}
    </text>
  );
}

export function PipelineArt({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={className}
      role="img"
      aria-label="Pipeline: a passive tap feeds a 45-dimensional host state into the learned dynamics model, which rolls out simulated futures that become a risk score with a lead time."
    >
      {/* the wire */}
      <line
        x1={C[0]}
        y1={Y}
        x2={C[3]}
        y2={Y}
        className="stroke-observed"
        strokeWidth="2"
      />

      {/* 1 — capture */}
      <path d={`M${C[0] - 4} ${Y} L${C[0] - 96} ${Y - 52} L${C[0] - 96} ${Y + 52} Z`} className="fill-node-lavender/35" />
      <g className="fill-node-blue/60 stroke-observed text-observed">
        <circle cx={C[0]} cy={Y} r="30" strokeWidth="2" />
        <path
          d={`M${C[0] - 12} ${Y + 6} L${C[0] - 4} ${Y - 4} L${C[0] + 3} ${Y + 3} L${C[0] + 12} ${Y - 9}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
      <Caption x={C[0]} label="PASSIVE TAP" />

      <Pill x={(C[0] + C[1]) / 2} label="HOST STATE (45-D)" />

      {/* 2 — the model: three instruments clustered on the wire */}
      {[
        { dx: -58, tone: "blue", text: "30s" },
        { dx: 0, tone: "lavender", text: "45-d" },
        { dx: 58, tone: "mint", text: "GRU" },
      ].map((n) => {
        const fill =
          n.tone === "blue"
            ? "fill-node-blue/60 stroke-observed text-observed"
            : n.tone === "lavender"
              ? "fill-node-lavender/60 stroke-projected text-projected"
              : "fill-node-mint/60 stroke-positive text-positive";
        return (
          <g key={n.dx} className={fill}>
            <circle cx={C[1] + n.dx} cy={Y} r="24" strokeWidth="2" />
            <text
              x={C[1] + n.dx}
              y={Y + 4}
              textAnchor="middle"
              fill="currentColor"
              stroke="none"
              fontSize="12"
              fontWeight="600"
            >
              {n.text}
            </text>
          </g>
        );
      })}
      <Caption x={C[1]} label="LEARNED DYNAMICS" />

      {/* 3 — the rollout fans out and comes back to one number */}
      <g className="stroke-threshold" fill="none" strokeWidth="2" strokeLinecap="round">
        {[-34, -12, 12, 34].map((k, i) => (
          <path
            key={k}
            d={`M${C[2] - 78} ${Y} Q${C[2]} ${Y + k * 1.5} ${C[2] + 78} ${Y}`}
            opacity={i === 1 || i === 2 ? 0.9 : 0.45}
            strokeDasharray={i === 1 || i === 2 ? undefined : "5 6"}
          />
        ))}
      </g>
      <circle cx={C[2] - 78} cy={Y} r="5" className="fill-threshold" />
      <circle cx={C[2] + 78} cy={Y} r="5" className="fill-threshold" />
      <Caption x={C[2]} label="K=6 ROLLOUT" />

      <Pill x={(C[2] + C[3]) / 2} label="SIMULATED FUTURES" />

      {/* 4 — the alert, with the same beam gesture as the tap */}
      <path d={`M${C[3] + 4} ${Y} L${C[3] + 96} ${Y - 52} L${C[3] + 96} ${Y + 52} Z`} className="fill-threshold/12" />
      <g className="fill-threshold/15 stroke-threshold text-threshold">
        <circle cx={C[3]} cy={Y} r="30" strokeWidth="2" />
        <path
          d={`M${C[3]} ${Y - 13} V${Y + 2}`}
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx={C[3]} cy={Y + 11} r="2.6" fill="currentColor" stroke="none" />
      </g>
      <Caption x={C[3]} label="RISK + LEAD TIME" />
    </svg>
  );
}
