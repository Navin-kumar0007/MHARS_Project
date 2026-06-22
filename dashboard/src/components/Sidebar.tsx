"use client";

import React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTelemetry } from "@/components/TelemetryProvider";
import { healthColor, healthLabel } from "@/components/ui";
import {
  LayoutDashboard,
  GitBranch,
  ScrollText,
  ChevronsLeftRightEllipsis,
  ChevronDown,
  BarChart3,
  Settings,
  LogOut,
  LogIn,
  Radio,
  Cpu,
  BookOpen,
  Activity,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard, perm: "view" },
  { href: "/analytics", label: "Analytics", icon: BarChart3, perm: "view" },
  { href: "/pipeline", label: "AI Pipeline", icon: GitBranch, perm: "view" },
  { href: "/history", label: "History", icon: ScrollText, perm: "view" },
  { href: "/diagnostics", label: "Diagnostics", icon: Activity, perm: "view" },
  { href: "/learn", label: "How It Works", icon: BookOpen, perm: "view" },
  { href: "/settings", label: "Settings", icon: Settings, perm: "view" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const {
    isConnected,
    latest,
    systemStatus,
    switchMachine,
    user,
    isAuthenticated,
    logout,
    can,
    liveMode,
    toggleMode,
  } = useTelemetry();
  const [machineOpen, setMachineOpen] = React.useState(false);

  const machines = systemStatus?.available_machines || {};
  // Canonical health: the composite digital-twin score (same number used everywhere).
  const health = (latest?.metadata?.health_score ?? latest?.system_health?.overall_score) as
    | number
    | undefined;
  const hColor = healthColor(health);

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col z-50 bg-[var(--surface)] backdrop-blur-xl border-r border-[var(--border)]">
      {/* Logo */}
      <div className="px-5 py-5">
        <div className="flex items-center gap-2.5">
          <div className="grid place-items-center w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-600 border border-indigo-400/40 shadow-[0_6px_16px_-6px_rgba(94,106,210,0.7)]">
            <ChevronsLeftRightEllipsis className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="text-lg font-bold text-gradient tracking-tight">
              MHARS
            </span>
            <p className="text-[10px] text-[var(--text-muted)] -mt-0.5 tracking-wider">DIGITAL TWIN · v2</p>
          </div>
        </div>
      </div>

      {/* Health summary */}
      <div className="px-4">
        <div className="rounded-xl bg-[var(--surface-3)] border border-[var(--border)] p-3">
          <div className="flex items-center justify-between">
            <span className="eyebrow">Machine Health</span>
            <span className="text-[11px]" style={{ color: hColor }}>
              {healthLabel(health)}
            </span>
          </div>
          <div className="flex items-end gap-1 mt-1.5">
            <span className="metric text-2xl" style={{ color: hColor }}>
              {health != null ? Math.round(health) : "—"}
            </span>
            <span className="text-xs text-[var(--text-muted)] mb-0.5">/ 100</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-[var(--surface-3)] overflow-hidden mt-2">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${health ?? 0}%`, background: hColor }}
            />
          </div>
        </div>
      </div>

      {/* Machine selector */}
      <div className="px-4 pt-3">
        <button
          onClick={() => setMachineOpen(!machineOpen)}
          disabled={!can("switch")}
          className="w-full flex items-center justify-between px-3 py-2 rounded-xl bg-[var(--surface-3)] border border-[var(--border)] hover:border-[var(--border-strong)] transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span className="flex items-center gap-2 text-[var(--text)] font-medium truncate">
            <Cpu className="w-3.5 h-3.5 text-[var(--text-dim)] shrink-0" />
            {latest?.machine_type || "Loading…"}
          </span>
          <ChevronDown className={`w-4 h-4 text-[var(--text-dim)] transition-transform ${machineOpen ? "rotate-180" : ""}`} />
        </button>

        {machineOpen && can("switch") && (
          <div className="mt-1.5 bg-[var(--surface)] border border-[var(--border)] rounded-xl overflow-hidden shadow-xl">
            {Object.entries(machines).map(([id, name]) => {
              const active = Number(id) === (latest?.machine_type_id ?? systemStatus?.machine_type_id);
              return (
                <button
                  key={id}
                  onClick={() => {
                    switchMachine(Number(id));
                    setMachineOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                    active ? "bg-indigo-400/10 text-indigo-600" : "text-[var(--text-dim)] hover:bg-[var(--surface-hover)] hover:text-[var(--text)]"
                  }`}
                >
                  {name as string}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 mt-1 space-y-0.5 overflow-y-auto">
        <p className="px-3 pb-2 eyebrow text-[var(--text-muted)]">Monitoring</p>
        {NAV_ITEMS.filter((item) => can(item.perm) || !isAuthenticated).map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                isActive
                  ? "text-indigo-700 bg-indigo-50 border border-indigo-100"
                  : "text-[var(--text-dim)] hover:text-[var(--text)] hover:bg-[var(--surface-hover)] border border-transparent"
              }`}
            >
              {isActive && <span className="absolute left-0 top-2 bottom-2 w-[3px] rounded-full bg-indigo-500" />}
              <item.icon className="w-4 h-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Data source: Demo / Live segmented toggle */}
      <div className="px-4 pb-3">
        <div className="flex items-center justify-between mb-1.5 px-0.5">
          <span className="eyebrow">Data Source</span>
          <span className="text-[10px]" style={{ color: liveMode ? "#f87171" : "#60a5fa" }}>
            {liveMode ? "real hardware" : "simulated"}
          </span>
        </div>
        {can("toggle_mode") ? (
          <div className="grid grid-cols-2 p-0.5 rounded-xl bg-[var(--surface-3)] border border-[var(--border)]">
            <button
              onClick={() => { if (liveMode) toggleMode(); }}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                !liveMode ? "bg-blue-500/20 text-blue-700" : "text-[var(--text-dim)] hover:text-[var(--text)]"
              }`}
            >
              <Radio className="w-3.5 h-3.5" /> Demo
            </button>
            <button
              onClick={() => { if (!liveMode) toggleMode(); }}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                liveMode ? "bg-rose-500/20 text-rose-700" : "text-[var(--text-dim)] hover:text-[var(--text)]"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" /> Live
            </button>
          </div>
        ) : (
          <div className={`flex items-center justify-center gap-1.5 py-1.5 rounded-xl text-xs font-semibold border ${
            liveMode ? "bg-rose-500/10 text-rose-700 border-rose-500/30" : "bg-blue-500/10 text-blue-700 border-blue-500/30"
          }`}>
            {liveMode ? <Cpu className="w-3.5 h-3.5" /> : <Radio className="w-3.5 h-3.5" />}
            {liveMode ? "Live" : "Demo"} <span className="text-[var(--text-muted)] font-normal">· operator only</span>
          </div>
        )}
      </div>

      {/* User / Connection */}
      <div className="p-4 border-t border-[var(--border)] space-y-3">
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full pulse-dot"
            style={{ color: isConnected ? "#34d399" : "#f87171", background: isConnected ? "#34d399" : "#f87171" }}
          />
          <span className="text-xs text-[var(--text-dim)]">{isConnected ? "Telemetry live" : "Disconnected"}</span>
          {latest && <span className="ml-auto text-[10px] text-[var(--text-muted)] metric">{latest.latency_ms.toFixed(1)}ms</span>}
        </div>

        {isAuthenticated ? (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="grid place-items-center w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400/30 to-indigo-500/30 text-indigo-700 text-xs font-bold shrink-0">
                {user?.username?.[0]?.toUpperCase() || "?"}
              </div>
              <div className="min-w-0">
                <p className="text-xs text-[var(--text)] font-medium truncate">{user?.username}</p>
                <p className="text-[10px] uppercase tracking-wider text-indigo-400/80">{user?.role}</p>
              </div>
            </div>
            <button
              onClick={() => {
                logout();
                router.push("/login");
              }}
              title="Log out"
              className="p-1.5 rounded-lg text-[var(--text-dim)] hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <Link
            href="/login"
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold text-indigo-600 bg-indigo-400/10 border border-indigo-400/30 hover:bg-indigo-400/20 transition-colors"
          >
            <LogIn className="w-3.5 h-3.5" />
            Sign In
          </Link>
        )}
      </div>
    </aside>
  );
}
