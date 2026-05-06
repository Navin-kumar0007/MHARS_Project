"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTelemetry } from "@/components/TelemetryProvider";
import {
  LayoutDashboard,
  GitBranch,
  ScrollText,
  Activity,
  ChevronDown,
} from "lucide-react";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/pipeline", label: "AI Pipeline", icon: GitBranch },
  { href: "/history", label: "History", icon: ScrollText },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { isConnected, latest, systemStatus, switchMachine } = useTelemetry();
  const [machineOpen, setMachineOpen] = React.useState(false);

  const machines = systemStatus?.available_machines || {};

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-[#08080C] border-r border-slate-800/60 flex flex-col z-50">
      {/* Logo */}
      <div className="p-5 border-b border-slate-800/60">
        <div className="flex items-center gap-2.5">
          <Activity className="w-6 h-6 text-teal-400" />
          <span className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-emerald-400">
            MHARS
          </span>
        </div>
        <p className="text-[10px] text-slate-600 mt-1 tracking-wider">
          DIGITAL TWIN v2.0
        </p>
      </div>

      {/* Machine Selector */}
      <div className="px-4 py-3 border-b border-slate-800/60">
        <button
          onClick={() => setMachineOpen(!machineOpen)}
          className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-slate-900/60 border border-slate-800/60 hover:border-slate-700 transition-colors text-sm"
        >
          <span className="text-slate-300 font-medium">
            {latest?.machine_type || "Loading..."}
          </span>
          <ChevronDown
            className={`w-4 h-4 text-slate-500 transition-transform ${
              machineOpen ? "rotate-180" : ""
            }`}
          />
        </button>

        {machineOpen && (
          <div className="mt-2 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
            {Object.entries(machines).map(([id, name]) => (
              <button
                key={id}
                onClick={() => {
                  switchMachine(Number(id));
                  setMachineOpen(false);
                }}
                className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                  Number(id) === (latest?.machine_type_id ?? systemStatus?.machine_type_id)
                    ? "bg-teal-500/10 text-teal-400"
                    : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`}
              >
                {name as string}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? "bg-teal-500/10 text-teal-400 border border-teal-500/20"
                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-900/50 border border-transparent"
              }`}
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Connection Status */}
      <div className="p-4 border-t border-slate-800/60">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              isConnected
                ? "bg-emerald-500 shadow-[0_0_6px_#10b981]"
                : "bg-rose-500 shadow-[0_0_6px_#f43f5e]"
            }`}
          />
          <span className="text-xs text-slate-500">
            {isConnected ? "Telemetry Live" : "Disconnected"}
          </span>
        </div>
        {latest && (
          <p className="text-[10px] text-slate-600 mt-1">
            Latency: {latest.latency_ms.toFixed(1)}ms
          </p>
        )}
      </div>
    </aside>
  );
}
