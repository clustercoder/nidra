import type { SVGProps } from "react";

/**
 * NIDRA brand mark: an observed trajectory that reaches the "now" rule and
 * fans out into projected futures. The mark is the product diagram.
 */
export function NidraMark({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden
      className={className}
      {...props}
    >
      <circle cx="32" cy="32" r="28" stroke="currentColor" strokeWidth="4" />
      <path
        d="M8 42c6.5 0 8.5-9 14.5-9 3.4 0 5.5 2 6.5 3.4"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
      />
      <path d="M29.5 20v24" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
      <g className="text-threshold" stroke="currentColor" strokeWidth="3.6" strokeLinecap="round">
        <path d="M31 35.5c5.5-6.5 9.5-10.5 14-13.5" />
        <path d="M31 35.5c6.5-2 12-2.5 17-2" />
        <path d="M31 35.5c5 4 9 7.5 12.5 12" />
      </g>
    </svg>
  );
}

/**
 * Full wordmark — mark + "NIDRA" set in the heading face.
 *
 * `className` drives BOTH size and colour: pass a text-size utility and the mark
 * scales with it (the SVG is sized in `em`), and pass a text-colour utility to
 * recolour it. Colour must not be baked in here — a hardcoded `text-navy` ties
 * with a caller's `text-gray-light` on specificity and wins on source order,
 * which is what made the mark invisible on the dark footer.
 */
export function NidraLogo({ className = "text-[22px] text-navy" }: { className?: string }) {
  return (
    <span
      className={`inline-flex items-center gap-[0.34em] transition-opacity active:opacity-50 lg:hover:opacity-80 ${className}`}
    >
      <NidraMark className="size-[1.45em] shrink-0" />
      <span className="font-headings font-bold tracking-wide">NIDRA</span>
    </span>
  );
}

/** GitHub mark — lucide dropped brand glyphs, so it lives here. */
export function GithubIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path
        fill="currentColor"
        d="M12 .5C5.73.5.98 5.24.98 11.5c0 4.86 3.15 8.98 7.52 10.44.55.1.75-.24.75-.53v-1.9c-3.06.66-3.71-1.3-3.71-1.3-.5-1.28-1.23-1.62-1.23-1.62-1-.68.08-.67.08-.67 1.1.08 1.69 1.14 1.69 1.14.98 1.69 2.58 1.2 3.21.92.1-.71.39-1.2.7-1.48-2.44-.28-5.01-1.22-5.01-5.45 0-1.2.43-2.19 1.14-2.96-.12-.28-.5-1.4.1-2.92 0 0 .93-.3 3.04 1.13a10.5 10.5 0 0 1 5.54 0c2.11-1.43 3.03-1.13 3.03-1.13.61 1.52.23 2.64.11 2.92.71.77 1.14 1.76 1.14 2.96 0 4.24-2.58 5.17-5.03 5.44.4.34.75 1.01.75 2.04v3.03c0 .29.2.64.76.53a10.53 10.53 0 0 0 7.51-10.44C23.02 5.24 18.27.5 12 .5Z"
      />
    </svg>
  );
}
