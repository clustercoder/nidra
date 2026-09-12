import { Badge, Beam, Cloud, Fan, Frame, HostNode } from "@/components/art/primitives";

/**
 * Hero illustration: traffic off a passive tap, a handful of hosts, and one
 * host's line read through a window — which carries on past the frame as a fan
 * of paths that have not happened yet. No axes and no numbers; the real chart
 * lives further down the page where it has room to be read.
 */
export function HeroArt({ className = "" }: { className?: string }) {
  // the observed line inside the frame ends here, and the fan picks it up
  const FRAME_X = 286;
  const FRAME_Y = 128;
  const FRAME_W = 232;
  const HAND_OFF_Y = 214;

  return (
    <svg
      viewBox="0 0 640 440"
      className={className}
      role="img"
      aria-label="Illustration: a passive tap reads traffic from three hosts; one host's line is shown inside a window and continues past it as a fan of projected paths, ending in a lead-time and an all-clear badge."
    >
      <Cloud x={96} y={64} s={1.6} />
      <Cloud x={556} y={78} s={1.1} />
      <Cloud x={196} y={392} s={1.3} />

      {/* the tap reads everything on the wire */}
      <Beam x={112} y={230} w={112} h={190} />
      <Badge x={88} y={230} r={30} tone="blue">
        <path
          d="M-11 5 L-4 -4 L2 3 L11 -9"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </Badge>
      <text x="88" y="284" textAnchor="middle" className="fill-gray-dark" fontSize="12">
        passive tap
      </text>

      {/* hosts on the segment */}
      <g>
        {[150, 230, 310].map((cy) => (
          <line
            key={cy}
            x1="224"
            y1={cy}
            x2="252"
            y2={cy}
            className="stroke-secondary-blue-light"
            strokeWidth="2"
          />
        ))}
        <line x1="243" y1="150" x2="243" y2="310" className="stroke-secondary-blue-light" strokeWidth="2" />
        <HostNode x={252} y={150} tone="idle" />
        <HostNode x={252} y={230} tone="alert" />
        <HostNode x={252} y={310} tone="idle" />
      </g>

      {/* the middle host's state, read through one window */}
      <Frame x={FRAME_X} y={FRAME_Y} w={FRAME_W} h={176}>
        <path
          d="M22 146 C62 142 84 136 112 118 C146 96 168 72 210 62"
          fill="none"
          className="stroke-observed"
          strokeWidth="3.5"
          strokeLinecap="round"
        />
        <line
          x1="168"
          y1="44"
          x2="168"
          y2="162"
          className="stroke-gray-medium"
          strokeWidth="2"
          strokeDasharray="4 5"
        />
        <text x="20" y="52" className="fill-gray-dark" fontSize="12">
          one host · last 15 min
        </text>
      </Frame>

      {/* what has not happened yet */}
      <Fan x={FRAME_X + FRAME_W - 22} y={HAND_OFF_Y} spread={64} len={130} />

      <Badge x={614} y={150} r={19} tone="amber">
        <path d="M0 -8 V1 L6 5" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
      </Badge>
      <Badge x={616} y={280} r={19} tone="mint">
        <path
          d="M-6 0 L-1 5 L7 -6"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </Badge>
      <text x="560" y="344" textAnchor="middle" className="fill-gray-dark" fontSize="12">
        next 3 minutes
      </text>
    </svg>
  );
}
