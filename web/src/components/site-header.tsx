"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, Menu } from "lucide-react";
import { NidraLogo } from "@/components/icons";
import { GITHUB_URL, NAV_LINKS } from "@/components/nav-data";

function AnnouncementBar() {
  return (
    <div className="hidden px-8 py-1 lg:block">
      <nav className="flex justify-end">
        <ul className="flex items-center gap-2.5">
          <li>
            <span className="text-gray-dark inline-flex items-center px-2 py-0.5 text-[15px]">
              SIH 2026 · Problem 26153
            </span>
          </li>
          <li className="border-gray-dark h-[10px] border-l" aria-hidden />
          <li>
            <a
              href={GITHUB_URL}
              className="text-gray-dark hover:text-primary-blue inline-flex items-center rounded-full border border-transparent px-2 py-0.5 text-[15px] transition-colors"
            >
              GitHub
            </a>
          </li>
        </ul>
      </nav>
    </div>
  );
}

export function SiteHeader() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 0);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <>
      <AnnouncementBar />
      <div className="sticky inset-x-0 -top-px z-50 transition-transform duration-300 ease-in-out">
        <div
          className={`relative border-b transition-all duration-300 ${
            scrolled
              ? "border-b-gray-light bg-white/80 backdrop-blur"
              : "border-b-transparent bg-white"
          }`}
        >
          <div className="flex h-16 items-center justify-between px-8">
            <div className="mr-4 flex justify-start lg:mr-0">
              <Link href="/" className="inline-block self-start">
                <span className="sr-only">NIDRA — home</span>
                <NidraLogo />
              </Link>
            </div>

            {/* mobile actions */}
            <div className="ml-auto flex items-center gap-4 lg:hidden">
              <button
                aria-label="Open navigation menu"
                className="text-gray-black flex items-center"
              >
                <Menu className="size-6" />
              </button>
            </div>

            {/* desktop nav */}
            <div className="hidden lg:flex lg:flex-1 lg:justify-center">
              <nav className="hidden w-full flex-1 items-center justify-center lg:flex">
                <ul className="flex list-none items-center justify-center gap-1">
                  {NAV_LINKS.map((link) => {
                    const active = link.href === "/";
                    return (
                      <li key={link.label}>
                        <a
                          href={link.href}
                          {...(link.external
                            ? { target: "_blank", rel: "noreferrer" }
                            : {})}
                          aria-current={active ? "page" : undefined}
                          className={`hover:text-primary-blue inline-flex w-max items-center border-b-2 px-4 py-2 text-base font-medium transition-colors ${
                            active
                              ? "border-primary-blue text-gray-black-soft"
                              : "text-gray-dark border-transparent"
                          }`}
                        >
                          {link.label}
                        </a>
                      </li>
                    );
                  })}
                </ul>
              </nav>
            </div>

            {/* desktop actions */}
            <div className="hidden items-center justify-end lg:flex lg:gap-5">
              <a
                href="/demo"
                className="group/button bg-primary-blue relative inline-flex items-center justify-start gap-1.5 rounded-full border border-transparent px-5 py-1.5 text-base font-medium text-white transition-all duration-200 hover:scale-[1.03] hover:bg-[rgb(24,64,180)]"
              >
                Run a forecast
                <ArrowRight className="size-4 transition-transform duration-200 group-hover/button:translate-x-0.5" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
