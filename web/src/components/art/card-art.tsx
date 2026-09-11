/**
 * Faint background motifs for the use-case cards. Each one echoes what the card
 * says; they sit at low opacity behind the copy and are purely decorative, so
 * they carry no label and no meaning the text does not already give.
 */
export type CardScene = "warning" | "headstart" | "airgap" | "unseen" | "replay";

function Warning() {
  // a curve crossing a threshold line, well before the end
  return (
    <>
      <line x1="0" y1="120" x2="300" y2="120" strokeDasharray="6 7" strokeWidth="2" />
      <path d="M6 196 C60 190 96 176 130 146 C164 116 190 78 236 54" fill="none" strokeWidth="3" />
      <circle cx="152" cy="120" r="9" strokeWidth="3" />
    </>
  );
}

function HeadStart() {
  // a clock, and the flows behind it
  return (
    <>
      <circle cx="150" cy="120" r="62" strokeWidth="3" fill="none" />
      <path d="M150 78 V122 L182 140" fill="none" strokeWidth="3" strokeLinecap="round" />
      {[34, 62, 90].map((y) => (
        <line key={y} x1="10" y1={y + 110} x2="94" y2={y + 110} strokeWidth="2" />
      ))}
    </>
  );
}

function AirGap() {
  // a stack behind a closed boundary
  return (
    <>
      {[0, 1, 2].map((i) => (
        <rect key={i} x="40" y={70 + i * 44} width="120" height="32" rx="7" strokeWidth="2.5" fill="none" />
      ))}
      <path d="M210 48 V212" strokeWidth="3" strokeDasharray="10 9" />
      <circle cx="210" cy="130" r="16" strokeWidth="2.5" fill="none" />
    </>
  );
}

function Unseen() {
  // one path the model never saw, dashed away from the rest
  return (
    <>
      <path d="M10 190 C70 184 104 166 140 132" fill="none" strokeWidth="3" />
      <path d="M140 132 C176 98 214 84 286 74" fill="none" strokeWidth="3" strokeDasharray="8 8" />
      <circle cx="140" cy="132" r="8" strokeWidth="3" fill="none" />
      <path d="M240 40 l14 14 -14 14" fill="none" strokeWidth="2.5" strokeLinecap="round" />
    </>
  );
}

function Replay() {
  // a scrub bar with markers, running fast
  return (
    <>
      <line x1="14" y1="130" x2="286" y2="130" strokeWidth="2.5" />
      {[46, 106, 166, 226].map((x, i) => (
        <circle key={x} cx={x} cy="130" r={i === 2 ? 11 : 6} strokeWidth="3" fill="none" />
      ))}
      <path d="M96 62 l26 24 -26 24" fill="none" strokeWidth="3" strokeLinecap="round" />
      <path d="M140 62 l26 24 -26 24" fill="none" strokeWidth="3" strokeLinecap="round" />
    </>
  );
}

export function CardArt({
  scene,
  className = "",
}: {
  scene: CardScene;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 300 240"
      className={className}
      fill="none"
      stroke="currentColor"
      aria-hidden
      focusable="false"
    >
      {scene === "warning" && <Warning />}
      {scene === "headstart" && <HeadStart />}
      {scene === "airgap" && <AirGap />}
      {scene === "unseen" && <Unseen />}
      {scene === "replay" && <Replay />}
    </svg>
  );
}
