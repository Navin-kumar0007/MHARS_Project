"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { API_BASE } from "@/components/TelemetryProvider";
import { Card, CardTitle, Badge, healthColor } from "@/components/ui";
import { ChevronsLeftRightEllipsis, Thermometer, HeartPulse, ShieldAlert, AlertTriangle } from "lucide-react";

type SharedStatus = {
  machine: string;
  current_temp: number | null;
  health_score: number | null;
  health_trend: string | null;
  action: string | null;
  alert: string | null;
  fault_type: string | null;
  thresholds: { idle: number; safe_max: number; critical: number } | null;
  timestamp: number | null;
};

export default function SharedViewPage() {
  const params = useParams();
  const token = Array.isArray(params.token) ? params.token[0] : (params.token as string);
  const [status, setStatus] = useState<SharedStatus | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) return;
    let stop = false;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/share/${token}`);
        if (res.status === 410 || res.status === 401) { setError("This share link is invalid or has expired."); return; }
        if (!res.ok) throw new Error(String(res.status));
        const data = await res.json();
        if (!stop) { setStatus(data); setError(""); }
      } catch {
        if (!stop) setError("Unable to reach the monitoring server.");
      }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => { stop = true; clearInterval(id); };
  }, [token]);

  const hColor = healthColor(status?.health_score ?? null);
  const temp = status?.current_temp;
  const th = status?.thresholds;
  const tColor = temp != null && th ? (temp >= th.critical ? "#f87171" : temp >= th.safe_max ? "#fbbf24" : "#34d399") : "#94a3b8";

  return (
    <div className="min-h-screen p-6 flex flex-col items-center">
      <div className="w-full max-w-3xl fade-in">
        <header className="flex items-center justify-between border-b border-white/[0.06] pb-4 mb-6">
          <div className="flex items-center gap-2.5">
            <div className="grid place-items-center w-9 h-9 rounded-xl bg-gradient-to-br from-teal-400/20 to-indigo-500/20 border border-teal-400/30">
              <ChevronsLeftRightEllipsis className="w-5 h-5 text-teal-300" />
            </div>
            <div>
              <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-300 to-indigo-300 tracking-tight">MHARS Status Report</h1>
              <p className="text-[10px] text-slate-500 tracking-wider">SHARED VIEW — READ ONLY</p>
            </div>
          </div>
          <Badge tone="neutral">{status?.machine || "—"}</Badge>
        </header>

        {error ? (
          <Card className="flex items-center gap-2 border-rose-500/30 bg-rose-500/5 text-rose-300">
            <ShieldAlert className="w-5 h-5" /> {error}
          </Card>
        ) : !status ? (
          <div className="text-center py-20 text-slate-600">Loading status…</div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Card>
                <CardTitle icon={Thermometer}>Temperature</CardTitle>
                <div className="metric text-4xl" style={{ color: tColor }}>{temp != null ? `${temp.toFixed(1)}°C` : "—"}</div>
                {th && <p className="text-[11px] text-slate-500 mt-2">Safe ≤ {th.safe_max}°C · Critical {th.critical}°C</p>}
              </Card>
              <Card>
                <CardTitle icon={HeartPulse}>Health Score</CardTitle>
                <div className="metric text-4xl" style={{ color: hColor }}>
                  {status.health_score != null ? Math.round(status.health_score) : "—"}
                  <span className="text-base text-slate-500 font-normal"> / 100</span>
                </div>
                <p className="text-[11px] text-slate-500 mt-2">Trend: {status.health_trend || "—"}</p>
              </Card>
            </div>

            <Card>
              <div className="flex items-center justify-between">
                <div>
                  <p className="eyebrow">Current Action</p>
                  <p className="metric text-2xl uppercase text-slate-100 mt-1">{status.action || "—"}</p>
                </div>
                <div className="text-right">
                  <p className="eyebrow">Fault Signature</p>
                  <p className="text-sm text-slate-300 mt-1">{status.fault_type || "Normal"}</p>
                </div>
              </div>
            </Card>

            <Card accent>
              <CardTitle icon={AlertTriangle}>Latest Alert</CardTitle>
              <div className="bg-black/30 rounded-xl p-3 border border-white/[0.06] font-mono text-xs leading-relaxed text-slate-300">
                {status.alert || "No alert."}
              </div>
            </Card>

            <p className="text-center text-[10px] text-slate-600">
              Auto-refreshing every 3s · Last update {status.timestamp ? new Date(status.timestamp * 1000).toLocaleTimeString() : "—"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
