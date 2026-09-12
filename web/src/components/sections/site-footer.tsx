import { GithubIcon, NidraLogo } from "@/components/icons";
import { GITHUB_URL } from "@/components/nav-data";

const CIC_URL = "https://www.unb.ca/cic/datasets/ids-2017.html";

const COLUMNS: { heading: string; links: { label: string; href: string }[] }[] = [
  {
    heading: "Product",
    links: [
      { label: "How it works", href: "#how-it-works" },
      { label: "Use cases", href: "#use-cases" },
      { label: "Demo", href: "/demo" },
      { label: "Benchmarks", href: "#evidence" },
    ],
  },
  {
    heading: "Resources",
    links: [
      { label: "Docs", href: `${GITHUB_URL}/tree/main/docs` },
      { label: "GitHub", href: GITHUB_URL },
      { label: "CIC-IDS2017 dataset", href: CIC_URL },
      { label: "PRD", href: `${GITHUB_URL}/blob/main/docs/PRD.pdf` },
    ],
  },
  {
    heading: "Project",
    links: [
      { label: "SIH 2026 · Problem 26153", href: "#evidence" },
      { label: "Team", href: GITHUB_URL },
      { label: "License", href: `${GITHUB_URL}/blob/main/LICENSE` },
    ],
  },
];

const LEGAL: { label: string; href: string }[] = [
  { label: "Status", href: "#evidence" },
  { label: "License", href: `${GITHUB_URL}/blob/main/LICENSE` },
  { label: "Evaluation methodology", href: `${GITHUB_URL}/tree/main/docs` },
];

export function SiteFooter() {
  return (
    <footer className="bg-gray-black py-24">
      <h2 className="sr-only">Footer</h2>
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="flex flex-col gap-16 xl:grid xl:grid-cols-[1fr_5fr_2fr] xl:gap-8">
          <div className="hidden md:block">
            <NidraLogo className="text-gray-light text-xl" />
          </div>
          <div className="grid grid-cols-2 gap-10 sm:grid-cols-3">
            {COLUMNS.map((col) => (
              <div key={col.heading}>
                <h3 className="text-gray-dark text-[15px] font-semibold tracking-wider uppercase">
                  {col.heading}
                </h3>
                <ul className="mt-4 space-y-2">
                  {col.links.map((link) => (
                    <li key={link.label}>
                      <a
                        href={link.href}
                        className="text-gray-light text-base transition-colors hover:text-white"
                      >
                        {link.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <div>
            <div className="mt-12 flex flex-col md:mt-0 xl:items-end">
              <div className="space-y-6 xl:space-y-8">
                <p className="text-gray-dark max-w-[280px] text-base leading-relaxed xl:text-right">
                  A predictive network world model. Built in the open for Smart
                  India Hackathon 2026.
                </p>
                <div className="flex space-x-6 xl:justify-end">
                  <a href={GITHUB_URL} aria-label="NIDRA on GitHub">
                    <GithubIcon className="text-gray-light size-5 transition-colors hover:text-white" />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="h-12 md:h-16 lg:h-24" />
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="border-gray-dark border-t" />
      </div>
      <div className="h-12 md:h-16 lg:h-24" />
      <div className="grid grid-cols-[1fr_min(1310px,calc(100%-4rem))_1fr] gap-x-8 [&>*]:col-start-2">
        <div className="flex flex-col-reverse justify-between gap-5 text-center md:flex-row md:gap-10">
          <p className="text-gray-light shrink-0 text-base">© 2026 NIDRA.</p>
          <p className="flex flex-wrap justify-center gap-5 md:justify-start">
            {LEGAL.map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="text-gray-light text-base transition-colors hover:text-white"
              >
                {item.label}
              </a>
            ))}
          </p>
        </div>
      </div>
    </footer>
  );
}
