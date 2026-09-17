"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowLeft, Bell, LayoutDashboard, Search, X } from "lucide-react";

import { NidraMark } from "@/components/icons";
import { ConsoleThemeToggle } from "@/components/console/theme-toggle";

export type ConsoleView = "dashboard" | "search";

/** The rail is the one Splunk signature that carries the whole redesign: a
 * persistent, icon-only nav standing in for the app's destinations, distinct
 * from the content it points at. Two real destinations only — a rail entry
 * with nothing behind it is worse than no rail. */
export function ConsoleRail({
  view,
  onViewChange,
  alertCount,
}: {
  view: ConsoleView;
  onViewChange: (v: ConsoleView) => void;
  alertCount: number;
}) {
  const items: { key: ConsoleView; label: string; Icon: typeof Search }[] = [
    { key: "dashboard", label: "Dashboard", Icon: LayoutDashboard },
    { key: "search", label: "Search", Icon: Search },
  ];
  return (
    <nav
      aria-label="Console navigation"
      className="border-console-line bg-console-surface flex w-14 shrink-0 flex-col items-center gap-1 border-r py-3"
    >
      <Link href="/" className="mb-2 flex size-9 items-center justify-center rounded-md">
        <NidraMark className="text-observed-lit size-6" />
      </Link>
      {items.map(({ key, label, Icon }) => {
        const active = view === key;
        return (
          <button
            key={key}
            onClick={() => onViewChange(key)}
            title={label}
            aria-current={active ? "true" : undefined}
            className={`relative flex size-10 items-center justify-center rounded-md transition-colors ${
              active
                ? "bg-console-raised text-observed-lit"
                : "text-console-muted hover:bg-console-raised/60 hover:text-console-text"
            }`}
          >
            {active && (
              <span className="bg-observed-lit absolute left-0 h-5 w-0.5 rounded-full" />
            )}
            <Icon className="size-[18px]" strokeWidth={2} />
            <span className="sr-only">{label}</span>
          </button>
        );
      })}
      <button
        onClick={() => onViewChange("search")}
        title="Notable events at or above threshold"
        className="text-console-muted hover:bg-console-raised/60 hover:text-console-text relative mt-1 flex size-10 items-center justify-center rounded-md transition-colors"
      >
        <Bell className="size-[18px]" strokeWidth={2} />
        {alertCount > 0 && (
          <span className="bg-threshold-lit text-console-bg absolute top-1.5 right-1.5 flex size-4 items-center justify-center rounded-full text-[9px] font-bold">
            {alertCount > 9 ? "9+" : alertCount}
          </span>
        )}
        <span className="sr-only">Alerts</span>
      </button>
    </nav>
  );
}

/** Page header — title, one line of what this view is for, and the controls
 * that apply to the whole view. Every reference console leads with this; the
 * console previously dropped the reader straight into panels. */
export function PageHeader({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <h1 className="font-headings text-console-text text-xl font-semibold">{title}</h1>
        <p className="text-console-muted mt-1 text-xs">{subtitle}</p>
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

export function ConsoleTopBar({
  breadcrumb,
  query,
  onQueryChange,
  modelLabel,
}: {
  breadcrumb: string;
  query: string;
  onQueryChange: (v: string) => void;
  modelLabel: string;
}) {
  return (
    <header className="border-console-line bg-console-surface/80 flex h-14 shrink-0 items-center gap-3 border-b px-4 backdrop-blur">
      <span className="text-console-muted hidden text-[11px] font-medium tracking-wide whitespace-nowrap sm:inline">
        NIDRA / <span className="text-console-text">{breadcrumb}</span>
      </span>
      <span className="bg-positive-lit/15 text-positive-lit hidden items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium sm:inline-flex">
        <span className="bg-positive-lit size-1.5 rounded-full" />
        replay live
      </span>

      <div className="relative mx-auto w-full max-w-md min-w-0 flex-1">
        <Search className="text-console-muted pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2" />
        <input
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Search hosts or stages…"
          spellCheck={false}
          className="border-console-line bg-console-raised/60 text-console-text placeholder:text-console-muted focus:border-observed-lit/60 focus:bg-console-raised w-full rounded-full border py-1.5 pr-8 pl-9 text-xs outline-none transition-colors"
        />
        {query && (
          <button
            onClick={() => onQueryChange("")}
            className="text-console-muted hover:text-console-text absolute top-1/2 right-2 -translate-y-1/2"
            aria-label="Clear search"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>

      <div className="text-console-muted flex items-center gap-4 text-xs">
        <span className="hidden font-mono md:inline">{modelLabel}</span>
        <ConsoleThemeToggle />
        <Link
          href="/"
          className="hover:text-console-text inline-flex items-center gap-1.5 whitespace-nowrap transition-colors"
        >
          <ArrowLeft className="size-4" />
          <span className="hidden sm:inline">Back to site</span>
        </Link>
      </div>
    </header>
  );
}
