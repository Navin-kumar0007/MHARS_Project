"use client";

import React from "react";
import { Moon, Sun } from "lucide-react";

export type Theme = "light" | "dark";
const KEY = "mhars_theme";

/** Read the theme the no-flash script already applied to <html>. */
function currentTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

/** Theme hook — mirrors <html> class, persists to localStorage, stays in sync across tabs. */
export function useTheme(): [Theme, (t: Theme) => void, () => void] {
  const [theme, setThemeState] = React.useState<Theme>("dark");

  React.useEffect(() => {
    setThemeState(currentTheme());
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY && (e.newValue === "light" || e.newValue === "dark")) apply(e.newValue);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const apply = React.useCallback((t: Theme) => {
    document.documentElement.classList.toggle("dark", t === "dark");
    try { localStorage.setItem(KEY, t); } catch {}
    setThemeState(t);
  }, []);

  const toggle = React.useCallback(() => apply(currentTheme() === "dark" ? "light" : "dark"), [apply]);
  return [theme, apply, toggle];
}

/** Segmented light/dark switch for the sidebar. */
export function ThemeToggle({ className = "" }: { className?: string }) {
  const [theme, , toggle] = useTheme();
  const isDark = theme === "dark";
  return (
    <button
      onClick={toggle}
      role="switch"
      aria-checked={isDark}
      aria-label={`Switch to ${isDark ? "light" : "dark"} theme`}
      title={`Switch to ${isDark ? "light" : "dark"} theme`}
      className={`relative flex items-center gap-1 p-0.5 rounded-full bg-[var(--surface-3)] border border-[var(--border)] hover:border-[var(--border-strong)] cursor-pointer ${className}`}
    >
      <span
        className={`grid place-items-center w-7 h-7 rounded-full transition-colors ${!isDark ? "bg-[var(--surface)] text-amber-500 shadow-[var(--sh-1)]" : "text-[var(--text-muted)]"}`}
      >
        <Sun className="w-3.5 h-3.5" />
      </span>
      <span
        className={`grid place-items-center w-7 h-7 rounded-full transition-colors ${isDark ? "bg-[var(--surface)] text-[var(--accent)] shadow-[var(--sh-1)]" : "text-[var(--text-muted)]"}`}
      >
        <Moon className="w-3.5 h-3.5" />
      </span>
    </button>
  );
}

/** Inline, render-blocking script → applies saved theme before first paint (no flash). Default: dark. */
export const themeScript = `(function(){try{var t=localStorage.getItem('${KEY}');if(t!=='light'){document.documentElement.classList.add('dark');}}catch(e){document.documentElement.classList.add('dark');}})();`;
