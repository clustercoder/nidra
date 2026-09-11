import { Badge, Cloud, Fan, HostNode } from "@/components/art/primitives";

/** One scene per carousel slide, in the same line-art language as the hero. */
export type SlideScene = "trajectory" | "backward" | "dynamics" | "forward";

const STAGES = ["recon", "foothold", "lateral", "exfil"];

function Trajectory() {
  const pts = [
    [40, 236],
    [140, 208],
    [240, 152],
    [340, 74],
  ];
  return (
    <>
      <Cloud x={330} y={40} s={1.1} />
      <path
        d={`M${pts[0][0]} ${pts[0][1]} C90 232 100 224 ${pts[1][0]} ${pts[1][1]} C190 196 200 178 ${pts[2][0]} ${pts[2][1]} C290 128 300 104 ${pts[3][0]} ${pts[3][1]}`}
        fill="none"
        className="stroke-observed"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      {pts.map(([cx, cy], i) => (
        <g key={i}>
          <circle
            cx={cx}
            cy={cy}
            r="13"
            fill="white"
            className={i === 3 ? "stroke-threshold" : "stroke-observed"}
            strokeWidth="2.5"
          />
          <circle
            cx={cx}
            cy={cy}
            r="5"
            className={i === 3 ? "fill-threshold" : "fill-observed"}
          />
          <text
            x={cx}
            y={cy + 34}
            textAnchor="middle"
            className="fill-gray-dark"
            fontSize="13"
          >
            {STAGES[i]}
          </text>
        </g>
      ))}
      <text x="40" y="286" className="fill-gray-dark" fontSize="13">
        minutes, not milliseconds
      </text>
    </>
  );
}

function Backward() {
  return (
    <>
      <Cloud x={340} y={48} s={1.1} />
      <line x1="36" y1="176" x2="368" y2="176" className="stroke-gray-medium" strokeWidth="2" />
      {[80, 160, 240, 320].map((cx, i) => (
        <g key={cx}>
          <line x1={cx} y1="168" x2={cx} y2="184" className="stroke-gray-medium" strokeWidth="2" />
          <circle
            cx={cx}
            cy="176"
            r="7"
            className={i < 2 ? "fill-gray-medium" : "fill-negative"}
          />
        </g>
      ))}
      {/* the event, and the alert arriving after it */}
      <text x="240" y="142" textAnchor="middle" className="fill-negative" fontSize="14" fontWeight="600">
        breach
      </text>
      <Badge x={320} y={100} r={26} tone="amber">
        <path d="M0 -11 V2 L8 7" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
      </Badge>
      <path
        d="M296 122 C270 148 258 156 246 166"
        fill="none"
        className="stroke-threshold"
        strokeWidth="2.5"
        strokeDasharray="5 6"
      />
      <text x="320" y="150" textAnchor="middle" className="fill-gray-dark" fontSize="13">
        alert
      </text>
      <text x="36" y="236" className="fill-gray-dark" fontSize="13">
        the signature fires after the fact
      </text>
    </>
  );
}

function Dynamics() {
  return (
    <>
      <Cloud x={60} y={48} s={1.1} />
      {[0, 1, 2].map((i) => (
        <g key={i} transform={`translate(${60 + i * 118} 150)`}>
          <rect
            x="-34"
            y="-46"
            width="68"
            height="92"
            rx="10"
            fill="white"
            className={i === 2 ? "stroke-projected" : "stroke-observed"}
            strokeWidth="2.5"
          />
          {[-28, -12, 4, 20].map((dy, k) => (
            <rect
              key={dy}
              x="-22"
              y={dy}
              width={[30, 42, 22, 36][k]}
              height="7"
              rx="3.5"
              className={i === 2 ? "fill-node-lavender" : "fill-node-blue"}
            />
          ))}
          <text y="66" textAnchor="middle" className="fill-gray-dark" fontSize="13">
            {i === 2 ? "t + 1" : `t − ${2 - i}`}
          </text>
        </g>
      ))}
      {[0, 1].map((i) => (
        <g key={i} className="stroke-gray-medium">
          <line x1={104 + i * 118} y1="150" x2={132 + i * 118} y2="150" strokeWidth="2" />
          <path d={`M${132 + i * 118} 150 l-8 -5 v10 z`} className="fill-gray-medium stroke-none" />
        </g>
      ))}
      <text x="60" y="252" className="fill-gray-dark" fontSize="13">
        learned from ordinary traffic
      </text>
    </>
  );
}

function Forward() {
  return (
    <>
      <Cloud x={70} y={44} s={1.1} />
      <path
        d="M30 214 C80 210 108 198 148 168"
        fill="none"
        className="stroke-observed"
        strokeWidth="3.5"
        strokeLinecap="round"
      />
      <line x1="148" y1="72" x2="148" y2="238" className="stroke-gray-medium" strokeWidth="2" strokeDasharray="4 5" />
      <text x="148" y="60" textAnchor="middle" className="fill-gray-dark" fontSize="13">
        now
      </text>
      <Fan x={148} y={168} spread={78} len={186} />
      <Badge x={366} y={96} r={22} tone="amber">
        <path d="M0 -9 V1 L7 5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      </Badge>
      <Badge x={368} y={242} r={22} tone="mint">
        <path d="M-7 0 L-2 6 L8 -7" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      </Badge>
      <text x="30" y="282" className="fill-gray-dark" fontSize="13">
        ~1000 of them, every 30 seconds
      </text>
      <HostNode x={30} y={214} tone="watch" />
    </>
  );
}

export function SlideArt({
  scene,
  className = "",
}: {
  scene: SlideScene;
  className?: string;
}) {
  const label: Record<SlideScene, string> = {
    trajectory: "A rising line through four waypoints labelled recon, foothold, lateral and exfil.",
    backward: "A timeline where the breach marker sits before the alert that follows it.",
    dynamics: "Three state vectors in sequence, the last one projected.",
    forward: "An observed line reaching now, then fanning into many projected paths.",
  };
  return (
    <svg
      viewBox="0 0 420 300"
      className={className}
      role="img"
      aria-label={label[scene]}
    >
      {scene === "trajectory" && <Trajectory />}
      {scene === "backward" && <Backward />}
      {scene === "dynamics" && <Dynamics />}
      {scene === "forward" && <Forward />}
    </svg>
  );
}
