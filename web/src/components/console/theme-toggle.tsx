"use client";

import * as React from "react";
import { Monitor, Moon, Sun } from "lucide-react";

export type ConsoleTheme = "dark" | "light" | "system";

export const THEME_KEY = "nidra-console-theme";

/* The stored choice lives outside React: it is read before paint by the script
   in the demo layout, and localStorage cannot be read during SSR. An external
   store keeps the control in step with the DOM without setting state from an
   effect. */
let listeners: (() => void)[] = [];

function subscribe(cb: () => void) {
  listeners.push(cb);
  return () => {
    listeners = listeners.filter((l) => l !== cb);
  };
}

function getSnapshot(): ConsoleTheme {
  const v = localStorage.getItem(THEME_KEY);
  return v === "light" || v === "dark" || v === "system" ? v : "dark";
}

/** Server and first client render agree on this; the store corrects after. */
function getServerSnapshot(): ConsoleTheme {
  return "dark";
}

/** The attribute the palette in globals.css keys off. */
function apply(theme: ConsoleTheme) {
  const resolved =
    theme === "system"
      ? window.matchMedia("(prefers-color-scheme: light)").matches
        ? "light"
        : "dark"
      : theme;
  document.documentElement.setAttribute("data-console-theme", resolved);
}

const OPTIONS: { value: ConsoleTheme; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "Match system", Icon: Monitor },
];

export function ConsoleThemeToggle() {
  const theme = React.useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot,
  );

  // Following the OS only matters while the user has not chosen for themselves.
  React.useEffect(() => {
    if (theme !== "system") return;
    const mql = window.matchMedia("(prefers-color-scheme: light)");
    const onChange = () => apply("system");
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [theme]);

  const choose = (next: ConsoleTheme) => {
    localStorage.setItem(THEME_KEY, next);
    apply(next);
    listeners.forEach((l) => l());
  };

  return (
    <div
      className="border-console-line bg-console-raised/60 flex items-center gap-0.5 rounded-lg border p-0.5"
      role="group"
      aria-label="Console theme"
    >
      {OPTIONS.map(({ value, label, Icon }) => (
        <button
          key={value}
          onClick={() => choose(value)}
          aria-pressed={theme === value}
          title={label}
          className={`flex size-7 items-center justify-center rounded-md transition-colors ${
            theme === value
              ? "bg-console-surface text-console-text"
              : "text-console-muted hover:text-console-text"
          }`}
        >
          <Icon className="size-3.5" strokeWidth={2} />
          <span className="sr-only">{label}</span>
        </button>
      ))}
    </div>
  );
}
