"use client";

import React, { useEffect, useState } from "react";
import { useTelemetry } from "@/components/TelemetryProvider";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import { ScrollText, Clock, AlertTriangle } from "lucide-react";

const API_BASE = "http://localhost:8050";

type ActionEntry = {
  timestamp: number;
  action: string;
  urgency: number;
  route: string;
  temp: number;
  latency_ms: number;
};

type AlertEntry = {
  timestamp: number;
  alert: string;
  source: string;
  urgency: number;
  severity: string;
};

function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function getActionBadge(action: string) {
  switch (action) {
    case "do-nothing":
      return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
    case "increase-fan":
    case "fan+":
      return "bg-amber-500/15 text-amber-400 border-amber-500/30";
    case "throttle":
      return "bg-orange-500/15 text-orange-400 border-orange-500/30";
    case "alert":
      return "bg-yellow-500/15 text-yellow-400 border-yellow-500/30";
    case "emergency-shutdown":
    case "shutdown":
      return "bg-rose-500/15 text-rose-400 border-rose-500/30";
    default:
      return "bg-slate-500/15 text-slate-400 border-slate-500/30";
  }
}

function getSeverityBadge(severity: string) {
  switch (severity) {
    case "critical":
      return "bg-rose-500/15 text-rose-400 border-rose-500/30";
    case "warning":
      return "bg-amber-500/15 text-amber-400 border-amber-500/30";
    default:
      return "bg-emerald-500/15 text-emerald-400 border-emerald-500/30";
  }
}

export default function HistoryPage() {
  const { history, latest } = useTelemetry();
  const [actions, setActions] = useState<ActionEntry[]>([]);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [activeTab, setActiveTab] = useState<"actions" | "alerts">("actions");

  // Poll history endpoints every 3 seconds
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const [actRes, alrtRes] = await Promise.all([
          fetch(`${API_BASE}/api/action_history`),
          fetch(`${API_BASE}/api/alert_history`),
        ]);
        const actData = await actRes.json();
        const alrtData = await alrtRes.json();
        setActions(actData.actions || []);
        setAlerts(alrtData.alerts || []);
      } catch (err) {
        console.error("Failed to fetch history", err);
      }
    };

    fetchHistory();
    const interval = setInterval(fetchHistory, 3000);
    return () => clearInterval(interval);
  }, []);

  // Full temp chart data
  const chartData = history.map((t) => ({
    time: timeLabel(t.timestamp),
    current_temp: t.current_temp,
    lstm_prediction: t.lstm_prediction,
    action: t.action,
  }));

  const thresholds = latest?.thresholds || { idle: 40, safe_max: 80, critical: 95 };

  return (
    <div className="p-6 space-y-6 relative overflow-hidden min-h-screen">
      {/* Background */}
      <div className="absolute bottom-[-20%] left-[-10%] w-[40%] h-[40%] bg-emerald-600/6 blur-[150px] rounded-full pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 border-b border-slate-800/60 pb-4">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
          <ScrollText className="w-6 h-6 text-emerald-400" />
          History &amp; Logs
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Complete timeline of RL agent actions, LLM alerts, and temperature history
        </p>
      </header>

      {/* Full Temperature Chart */}
      <div className="relative z-10 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Full Temperature History (Last 2 Minutes)
        </h2>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="time" stroke="#475569" fontSize={10} tickMargin={6} />
              <YAxis stroke="#475569" fontSize={10} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#1e293b",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                itemStyle={{ color: "#e2e8f0" }}
              />
              <Legend iconType="circle" wrapperStyle={{ fontSize: "11px" }} />
              <ReferenceLine
                y={thresholds.safe_max}
                stroke="#f59e0b"
                strokeDasharray="4 4"
                label={{
                  value: "Safe Max",
                  fill: "#f59e0b",
                  fontSize: 10,
                  position: "right",
                }}
              />
              <ReferenceLine
                y={thresholds.critical}
                stroke="#ef4444"
                strokeDasharray="4 4"
                label={{
                  value: "Critical",
                  fill: "#ef4444",
                  fontSize: 10,
                  position: "right",
                }}
              />
              <Line
                type="monotone"
                dataKey="current_temp"
                name="Actual Temp (°C)"
                stroke="#f43f5e"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="lstm_prediction"
                name="LSTM Prediction"
                stroke="#14b8a6"
                strokeWidth={1.5}
                strokeDasharray="5 5"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Tabs: Actions vs Alerts */}
      <div className="relative z-10">
        <div className="flex gap-1 mb-4">
          <button
            onClick={() => setActiveTab("actions")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "actions"
                ? "bg-teal-500/10 text-teal-400 border border-teal-500/30"
                : "text-slate-500 hover:text-slate-300 border border-transparent"
            }`}
          >
            <Clock className="w-4 h-4 inline mr-2" />
            Action Timeline ({actions.length})
          </button>
          <button
            onClick={() => setActiveTab("alerts")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === "alerts"
                ? "bg-indigo-500/10 text-indigo-400 border border-indigo-500/30"
                : "text-slate-500 hover:text-slate-300 border border-transparent"
            }`}
          >
            <AlertTriangle className="w-4 h-4 inline mr-2" />
            Alert Log ({alerts.length})
          </button>
        </div>

        {/* Action Timeline */}
        {activeTab === "actions" && (
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl shadow-xl overflow-hidden">
            <div className="max-h-[400px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-900 border-b border-slate-800">
                  <tr className="text-xs text-slate-500 uppercase tracking-wider">
                    <th className="text-left px-5 py-3">Time</th>
                    <th className="text-left px-5 py-3">Action</th>
                    <th className="text-left px-5 py-3">Urgency</th>
                    <th className="text-left px-5 py-3">Route</th>
                    <th className="text-left px-5 py-3">Temp</th>
                    <th className="text-right px-5 py-3">Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {[...actions].reverse().map((entry, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                    >
                      <td className="px-5 py-3 font-mono text-xs text-slate-400">
                        {timeLabel(entry.timestamp)}
                      </td>
                      <td className="px-5 py-3">
                        <span
                          className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border uppercase ${getActionBadge(
                            entry.action
                          )}`}
                        >
                          {entry.action}
                        </span>
                      </td>
                      <td className="px-5 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${entry.urgency * 100}%`,
                                background:
                                  entry.urgency > 0.8
                                    ? "#ef4444"
                                    : entry.urgency > 0.5
                                    ? "#f59e0b"
                                    : "#6366f1",
                              }}
                            />
                          </div>
                          <span className="text-xs text-slate-400 font-mono">
                            {entry.urgency.toFixed(3)}
                          </span>
                        </div>
                      </td>
                      <td className="px-5 py-3">
                        <span
                          className={`text-xs font-semibold uppercase ${
                            entry.route === "edge"
                              ? "text-teal-400"
                              : entry.route === "cloud"
                              ? "text-blue-400"
                              : "text-purple-400"
                          }`}
                        >
                          {entry.route}
                        </span>
                      </td>
                      <td className="px-5 py-3 font-mono text-xs text-slate-300">
                        {entry.temp.toFixed(1)}°C
                      </td>
                      <td className="px-5 py-3 text-right font-mono text-xs text-slate-500">
                        {entry.latency_ms.toFixed(1)}ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {actions.length === 0 && (
                <div className="text-center py-12 text-slate-600 text-sm">
                  No actions recorded yet. Telemetry data will appear here automatically.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Alert Log */}
        {activeTab === "alerts" && (
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl shadow-xl overflow-hidden">
            <div className="max-h-[400px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-900 border-b border-slate-800">
                  <tr className="text-xs text-slate-500 uppercase tracking-wider">
                    <th className="text-left px-5 py-3">Time</th>
                    <th className="text-left px-5 py-3">Severity</th>
                    <th className="text-left px-5 py-3">Source</th>
                    <th className="text-left px-5 py-3">Alert</th>
                  </tr>
                </thead>
                <tbody>
                  {[...alerts].reverse().map((entry, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors"
                    >
                      <td className="px-5 py-3 font-mono text-xs text-slate-400 whitespace-nowrap">
                        {timeLabel(entry.timestamp)}
                      </td>
                      <td className="px-5 py-3">
                        <span
                          className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold border uppercase ${getSeverityBadge(
                            entry.severity
                          )}`}
                        >
                          {entry.severity}
                        </span>
                      </td>
                      <td className="px-5 py-3">
                        <span className="text-xs bg-slate-800/60 text-slate-300 px-2 py-0.5 rounded border border-slate-700/60">
                          {entry.source}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-xs text-slate-300 max-w-md">
                        {entry.alert}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {alerts.length === 0 && (
                <div className="text-center py-12 text-slate-600 text-sm">
                  No alerts generated yet. Alerts will appear here as the LLM processes telemetry.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
