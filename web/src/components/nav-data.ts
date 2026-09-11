export interface NavLink {
  label: string;
  href: string;
  /** Opens in a new tab. Set for links that leave the marketing site. */
  external?: boolean;
}

export const GITHUB_URL = "https://github.com/clustercoder/nidra";

export const NAV_LINKS: NavLink[] = [
  { label: "Home", href: "/" },
  { label: "How it works", href: "#how-it-works" },
  { label: "Use cases", href: "#use-cases" },
  { label: "Docs", href: GITHUB_URL, external: true },
];
