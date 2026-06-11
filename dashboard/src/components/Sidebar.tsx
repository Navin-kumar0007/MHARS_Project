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
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: LayoutDashboard, perm: "view" },
  { href: "/analytics", label: "Analytics", icon: BarChart3, perm: "view" },
  { href: "/pipeline", label: "AI Pipeline", icon: GitBranch, perm: "view" },
  { href: "/history", label: "History", icon: ScrollText, perm: "view" },
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
    <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col z-50 bg-[#080b13]/95 backdrop-blur-xl border-r border-white/[0.06]">
      {/* Logo */}
      <div className="px-5 py-5">
        <div className="flex items-center gap-2.5">
          <div className="grid place-items-center w-9 h-9 rounded-xl bg-gradient-to-br from-teal-400/20 to-indigo-500/20 border border-teal-400/30">
            <ChevronsLeftRightEllipsis className="w-5 h-5 text-teal-300" />
          </div>
          <div>
            <span className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-300 to-indigo-300 tracking-tight">
              MHARS
            </span>
            <p className="text-[10px] text-slate-600 -mt-0.5 tracking-wider">DIGITAL TWIN · v2</p>
          </div>
        </div>
      </div>

      {/* Health summary */}
      <div className="px-4">
        <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3">
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
            <span className="text-xs text-slate-600 mb-0.5">/ 100</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-white/[0.06] overflow-hidden mt-2">
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
          className="w-full flex items-center justify-between px-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:border-white/[0.14] transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <span className="flex items-center gap-2 text-slate-200 font-medium truncate">
            <Cpu className="w-3.5 h-3.5 text-slate-500 shrink-0" />
            {latest?.machine_type || "Loading…"}
          </span>
          <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${machineOpen ? "rotate-180" : ""}`} />
        </button>

        {machineOpen && can("switch") && (
          <div className="mt-1.5 bg-[#0c1120] border border-white/[0.08] rounded-xl overflow-hidden shadow-xl">
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
                    active ? "bg-teal-400/10 text-teal-300" : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-200"
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
        <p className="px-3 pb-2 eyebrow text-slate-600">Monitoring</p>
        {NAV_ITEMS.filter((item) => can(item.perm) || !isAuthenticated).map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${
                isActive ? "text-teal-300 bg-teal-400/10" : "text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]"
              }`}
            >
              {isActive && <span className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full bg-teal-300" />}
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
          <div className="grid grid-cols-2 p-0.5 rounded-xl bg-white/[0.04] border border-white/[0.06]">
            <button
              onClick={() => { if (liveMode) toggleMode(); }}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                !liveMode ? "bg-blue-500/20 text-blue-200" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              <Radio className="w-3.5 h-3.5" /> Demo
            </button>
            <button
              onClick={() => { if (!liveMode) toggleMode(); }}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                liveMode ? "bg-rose-500/20 text-rose-200" : "text-slate-500 hover:text-slate-300"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" /> Live
            </button>
          </div>
        ) : (
          <div className={`flex items-center justify-center gap-1.5 py-1.5 rounded-xl text-xs font-semibold border ${
            liveMode ? "bg-rose-500/10 text-rose-300 border-rose-500/30" : "bg-blue-500/10 text-blue-300 border-blue-500/30"
          }`}>
            {liveMode ? <Cpu className="w-3.5 h-3.5" /> : <Radio className="w-3.5 h-3.5" />}
            {liveMode ? "Live" : "Demo"} <span className="text-slate-600 font-normal">· operator only</span>
          </div>
        )}
      </div>

      {/* User / Connection */}
      <div className="p-4 border-t border-white/[0.06] space-y-3">
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full pulse-dot"
            style={{ color: isConnected ? "#34d399" : "#f87171", background: isConnected ? "#34d399" : "#f87171" }}
          />
          <span className="text-xs text-slate-500">{isConnected ? "Telemetry live" : "Disconnected"}</span>
          {latest && <span className="ml-auto text-[10px] text-slate-600 metric">{latest.latency_ms.toFixed(1)}ms</span>}
        </div>

        {isAuthenticated ? (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="grid place-items-center w-8 h-8 rounded-full bg-gradient-to-br from-teal-400/30 to-indigo-500/30 text-teal-200 text-xs font-bold shrink-0">
                {user?.username?.[0]?.toUpperCase() || "?"}
              </div>
              <div className="min-w-0">
                <p className="text-xs text-slate-200 font-medium truncate">{user?.username}</p>
                <p className="text-[10px] uppercase tracking-wider text-teal-400/80">{user?.role}</p>
              </div>
            </div>
            <button
              onClick={() => {
                logout();
                router.push("/login");
              }}
              title="Log out"
              className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <Link
            href="/login"
            className="flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-semibold text-teal-300 bg-teal-400/10 border border-teal-400/30 hover:bg-teal-400/20 transition-colors"
          >
            <LogIn className="w-3.5 h-3.5" />
            Sign In
          </Link>
        )}
      </div>
    </aside>
  );
}
