/**
 * Line-art primitives for the marketing illustrations.
 *
 * One visual language: thin strokes, mostly-white fills, a navy/blue line with
 * amber reserved for anything projected forward. Compose these rather than
 * hand-drawing each scene, so the hero and the carousel stay consistent.
 */

export function Cloud({
  x,
  y,
  s = 1,
  className = "stroke-secondary-blue-light",
}: {
  x: number;
  y: number;
  s?: number;
  className?: string;
}) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`} className={className}>
      <path
        d="M0 34c-9 0-16-7-16-15S-9 4 0 4c2-9 10-15 19-15s17 6 19 15c2-1 4-2 7-2 9 0 16 7 16 16s-7 16-16 16H0Z"
        fill="white"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </g>
  );
}

/** A host on the network: rounded square with a status dot. */
export function HostNode({
  x,
  y,
  label,
  tone = "idle",
}: {
  x: number;
  y: number;
  label?: string;
  tone?: "idle" | "watch" | "alert";
}) {
  const fill =
    tone === "alert"
      ? "fill-threshold/15"
      : tone === "watch"
        ? "fill-node-lavender/50"
        : "fill-node-blue/40";
  const stroke =
    tone === "alert"
      ? "stroke-threshold"
      : tone === "watch"
        ? "stroke-projected"
        : "stroke-observed";
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect
        x="-17"
        y="-14"
        width="34"
        height="28"
        rx="7"
        className={`${fill} ${stroke}`}
        strokeWidth="2"
      />
      <line x1="-9" y1="-5" x2="9" y2="-5" className={stroke} strokeWidth="2" strokeLinecap="round" />
      <line x1="-9" y1="1" x2="3" y2="1" className={stroke} strokeWidth="2" strokeLinecap="round" />
      <circle cx="9" cy="7" r="2.5" className={stroke.replace("stroke-", "fill-")} />
      {label && (
        <text y="30" textAnchor="middle" className="fill-gray-dark" fontSize="10">
          {label}
        </text>
      )}
    </g>
  );
}

/** Small circular badge holding a glyph — the little accents in the scene. */
export function Badge({
  x,
  y,
  r = 15,
  tone = "blue",
  children,
}: {
  x: number;
  y: number;
  r?: number;
  tone?: "blue" | "lavender" | "amber" | "mint";
  children?: React.ReactNode;
}) {
  const map = {
    blue: "fill-node-blue/60 stroke-observed text-observed",
    lavender: "fill-node-lavender/60 stroke-projected text-projected",
    amber: "fill-threshold/15 stroke-threshold text-threshold",
    mint: "fill-node-mint/60 stroke-positive text-positive",
  } as const;
  return (
    <g transform={`translate(${x} ${y})`} className={map[tone]}>
      <circle r={r} strokeWidth="2" />
      {children}
    </g>
  );
}

/** The signature gesture: the observed line reaching now, then fanning forward. */
export function Fan({
  x,
  y,
  spread = 46,
  len = 150,
  className = "stroke-threshold",
}: {
  x: number;
  y: number;
  spread?: number;
  len?: number;
  className?: string;
}) {
  const arcs = [-1, -0.35, 0.35, 1];
  return (
    <g className={className} fill="none" strokeWidth="2.5" strokeLinecap="round">
      {arcs.map((k, i) => (
        <path
          key={i}
          d={`M${x} ${y} Q${x + len * 0.5} ${y + k * spread * 0.45} ${x + len} ${y + k * spread}`}
          opacity={i === 1 || i === 2 ? 1 : 0.55}
          strokeDasharray={i === 1 || i === 2 ? undefined : "5 6"}
        />
      ))}
      <circle cx={x} cy={y} r="4" className="fill-threshold stroke-none" />
    </g>
  );
}

/** A soft sweep of light, used for the capture/scan gesture. */
export function Beam({
  x,
  y,
  w = 120,
  h = 70,
  flip = false,
}: {
  x: number;
  y: number;
  w?: number;
  h?: number;
  flip?: boolean;
}) {
  const d = flip
    ? `M0 0 L${-w} ${-h / 2} L${-w} ${h / 2} Z`
    : `M0 0 L${w} ${-h / 2} L${w} ${h / 2} Z`;
  return (
    <g transform={`translate(${x} ${y})`}>
      <path d={d} className="fill-node-lavender/35" />
    </g>
  );
}

/** Window/screen frame — the thing being observed. */
export function Frame({
  x,
  y,
  w,
  h,
  children,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  children?: React.ReactNode;
}) {
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect
        width={w}
        height={h}
        rx="16"
        fill="white"
        className="stroke-observed"
        strokeWidth="2.5"
      />
      <line
        x1="0"
        y1="30"
        x2={w}
        y2="30"
        className="stroke-secondary-blue-light"
        strokeWidth="2"
      />
      {[16, 30, 44].map((cx) => (
        <circle key={cx} cx={cx} cy="15" r="3.5" className="fill-secondary-blue-light" />
      ))}
      {children}
    </g>
  );
}
