"use client";

import React, { useEffect, useState } from "react";
import { useTelemetry, API_BASE } from "@/components/TelemetryProvider";
import {
  LineChart,
  Line,
  Area,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import { ScrollText, Clock, AlertTriangle } from "lucide-react";
import { Card, CardTitle, PageHeader, Badge, CHART, tooltipStyle, tooltipLabelStyle } from "@/components/ui";

type ActionEntry = { timestamp: number; action: string; urgency: number; route: string; temp: number; latency_ms: number };
type AlertEntry = { timestamp: number; alert: string; source: string; urgency: number; severity: string };

function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function actionTone(action: string): string {
  switch (action) {
    case "do-nothing": return "good";
    case "increase-fan":
    case "fan+":
    case "alert": return "warn";
    case "throttle": return "warn";
    case "emergency-shutdown":
    case "shutdown": return "bad";
    default: return "neutral";
  }
}

export default function HistoryPage() {
  const { history, latest } = useTelemetry();
  const [actions, setActions] = useState<ActionEntry[]>([]);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [activeTab, setActiveTab] = useState<"actions" | "alerts">("actions");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const [actRes, alrtRes] = await Promise.all([
          fetch(`${API_BASE}/api/action_history`),
          fetch(`${API_BASE}/api/alert_history`),
        ]);
        setActions((await actRes.json()).actions || []);
        setAlerts((await alrtRes.json()).alerts || []);
      } catch (err) {
        console.error("Failed to fetch history", err);
      }
    };
    fetchHistory();
    const interval = setInterval(fetchHistory, 3000);
    return () => clearInterval(interval);
  }, []);

  const chartData = history.map((t) => ({ time: timeLabel(t.timestamp), current_temp: t.current_temp, lstm_prediction: t.lstm_prediction }));
  const thresholds = latest?.thresholds || { idle: 40, safe_max: 80, critical: 95 };

  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      <PageHeader icon={ScrollText} title="History & Logs" subtitle="Timeline of RL actions, LLM alerts and temperature" accent="#34d399" />

      <Card>
        <CardTitle>Full Temperature History (last ~2 minutes)</CardTitle>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 6, right: 12, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="histTemp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CHART.temp} stopOpacity={0.16} />
                  <stop offset="100%" stopColor={CHART.temp} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="time" stroke={CHART.axis} fontSize={10} tickMargin={6} tickLine={false} axisLine={false} />
              <YAxis stroke={CHART.axis} fontSize={10} domain={["auto", "auto"]} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={{ color: "#e2e8f0" }} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine y={thresholds.safe_max} stroke={CHART.warn} strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "Safe Max", fill: CHART.warn, fontSize: 10, position: "right" }} />
              <ReferenceLine y={thresholds.critical} stroke={CHART.crit} strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "Critical", fill: CHART.crit, fontSize: 10, position: "right" }} />
              <Area type="monotone" dataKey="current_temp" name="Actual Temp (°C)" stroke={CHART.temp} strokeWidth={2} fill="url(#histTemp)" dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="lstm_prediction" name="LSTM Prediction" stroke={CHART.teal} strokeWidth={1.5} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <div>
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setActiveTab("actions")}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${activeTab === "actions" ? "bg-teal-400/10 text-teal-300 border border-teal-400/30" : "text-slate-500 hover:text-slate-300 border border-transparent"}`}
          >
            <Clock className="w-4 h-4 inline mr-2" /> Action Timeline ({actions.length})
          </button>
          <button
            onClick={() => setActiveTab("alerts")}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${activeTab === "alerts" ? "bg-indigo-400/10 text-indigo-300 border border-indigo-400/30" : "text-slate-500 hover:text-slate-300 border border-transparent"}`}
          >
            <AlertTriangle className="w-4 h-4 inline mr-2" /> Alert Log ({alerts.length})
          </button>
        </div>

        {activeTab === "actions" && (
          <Card className="!p-0 overflow-hidden">
            <div className="max-h-[420px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-[#0c1120] border-b border-white/[0.08] z-10">
                  <tr className="eyebrow">
                    <th className="text-left px-5 py-3 font-semibold">Time</th>
                    <th className="text-left px-5 py-3 font-semibold">Action</th>
                    <th className="text-left px-5 py-3 font-semibold">Urgency</th>
                    <th className="text-left px-5 py-3 font-semibold">Route</th>
                    <th className="text-left px-5 py-3 font-semibold">Temp</th>
                    <th className="text-right px-5 py-3 font-semibold">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {[...actions].reverse().map((entry, idx) => (
                    <tr key={idx} className="border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors">
                      <td className="px-5 py-3 metric text-xs text-slate-400">{timeLabel(entry.timestamp)}</td>
                      <td className="px-5 py-3"><Badge tone={actionTone(entry.action)}>{entry.action}</Badge></td>
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-white/[0.08] rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${entry.urgency * 100}%`, background: entry.urgency > 0.8 ? CHART.crit : entry.urgency > 0.5 ? CHART.warn : CHART.indigo }} />
                          </div>
                          <span className="metric text-xs text-slate-400">{entry.urgency.toFixed(3)}</span>
                        </div>
                      </td>
                      <td className="px-5 py-3"><span className={`text-xs font-semibold uppercase ${entry.route === "edge" ? "text-cyan-300" : entry.route === "cloud" ? "text-blue-300" : "text-indigo-300"}`}>{entry.route}</span></td>
                      <td className="px-5 py-3 metric text-xs text-slate-300">{entry.temp.toFixed(1)}°C</td>
                      <td className="px-5 py-3 text-right metric text-xs text-slate-500">{entry.latency_ms.toFixed(1)}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {actions.length === 0 && <div className="text-center py-12 text-slate-600 text-sm">No actions recorded yet.</div>}
            </div>
          </Card>
        )}

        {activeTab === "alerts" && (
          <Card className="!p-0 overflow-hidden">
            <div className="max-h-[420px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-[#0c1120] border-b border-white/[0.08] z-10">
                  <tr className="eyebrow">
                    <th className="text-left px-5 py-3 font-semibold">Time</th>
                    <th className="text-left px-5 py-3 font-semibold">Severity</th>
                    <th className="text-left px-5 py-3 font-semibold">Source</th>
                    <th className="text-left px-5 py-3 font-semibold">Alert</th>
                  </tr>
                </thead>
                <tbody>
                  {[...alerts].reverse().map((entry, idx) => (
                    <tr key={idx} className="border-b border-white/[0.04] hover:bg-white/[0.03] transition-colors">
                      <td className="px-5 py-3 metric text-xs text-slate-400 whitespace-nowrap">{timeLabel(entry.timestamp)}</td>
                      <td className="px-5 py-3"><Badge tone={entry.severity === "critical" ? "bad" : entry.severity === "warning" ? "warn" : "good"}>{entry.severity}</Badge></td>
                      <td className="px-5 py-3"><span className="text-xs bg-white/[0.04] text-slate-300 px-2 py-0.5 rounded border border-white/[0.08]">{entry.source}</span></td>
                      <td className="px-5 py-3 text-xs text-slate-300 max-w-md">{entry.alert}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {alerts.length === 0 && <div className="text-center py-12 text-slate-600 text-sm">No alerts generated yet.</div>}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
