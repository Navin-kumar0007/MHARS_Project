"use client";

import React from "react";
import { useTelemetry } from "@/components/TelemetryProvider";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";
import {
  Cpu,
  Fan,
  Wifi,
  ShieldAlert,
  Zap,
  Flame,
  Settings,
  Wind,
  Radio,
  BatteryWarning,
  RotateCcw,
  Network,
} from "lucide-react";

// ── Helper: format time label ─────────────────────────────────────────────────
function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
  });
}

// ── Helper: action color mapping ──────────────────────────────────────────────
function getActionStyle(action: string) {
  switch (action) {
    case "do-nothing":
      return "text-emerald-400 border-emerald-500/30 bg-emerald-500/10";
    case "increase-fan":
    case "fan+":
      return "text-amber-400 border-amber-500/30 bg-amber-500/10";
    case "throttle":
      return "text-orange-400 border-orange-500/30 bg-orange-500/10";
    case "alert":
      return "text-yellow-400 border-yellow-500/30 bg-yellow-500/10";
    case "emergency-shutdown":
    case "shutdown":
      return "text-rose-500 border-rose-500/30 bg-rose-500/10";
    default:
      return "text-gray-400 border-gray-500/30 bg-gray-500/10";
  }
}

// ── Temperature Gauge (SVG) ───────────────────────────────────────────────────
function TemperatureGauge({
  value,
  idle,
  safeMax,
  critical,
}: {
  value: number;
  idle: number;
  safeMax: number;
  critical: number;
}) {
  // Map temperature to angle (0-180 degrees)
  const minTemp = Math.max(idle - 10, 0);
  const maxTemp = critical + 15;
  const clampedValue = Math.max(minTemp, Math.min(maxTemp, value));
  const ratio = (clampedValue - minTemp) / (maxTemp - minTemp);
  const angle = ratio * 180;

  // Zone boundaries in degrees
  const safeAngle = ((safeMax - minTemp) / (maxTemp - minTemp)) * 180;
  const critAngle = ((critical - minTemp) / (maxTemp - minTemp)) * 180;

  // Needle color
  const needleColor =
    value >= critical
      ? "#ef4444"
      : value >= safeMax
      ? "#f59e0b"
      : "#10b981";

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" className="w-full max-w-[260px]">
        {/* Green zone (idle to safe) */}
        <path
          d={describeArc(100, 100, 80, 180, 180 + safeAngle)}
          fill="none"
          stroke="#10b981"
          strokeWidth="12"
          strokeLinecap="round"
          opacity="0.3"
        />
        {/* Amber zone (safe to critical) */}
        <path
          d={describeArc(100, 100, 80, 180 + safeAngle, 180 + critAngle)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth="12"
          strokeLinecap="round"
          opacity="0.3"
        />
        {/* Red zone (above critical) */}
        <path
          d={describeArc(100, 100, 80, 180 + critAngle, 360)}
          fill="none"
          stroke="#ef4444"
          strokeWidth="12"
          strokeLinecap="round"
          opacity="0.3"
        />
        {/* Needle */}
        <line
          x1="100"
          y1="100"
          x2={100 + 65 * Math.cos((Math.PI * (180 + angle)) / 180)}
          y2={100 + 65 * Math.sin((Math.PI * (180 + angle)) / 180)}
          stroke={needleColor}
          strokeWidth="3"
          strokeLinecap="round"
          style={{
            filter: `drop-shadow(0 0 6px ${needleColor})`,
            transition: "all 0.5s ease-out",
          }}
        />
        {/* Center dot */}
        <circle cx="100" cy="100" r="5" fill={needleColor} opacity="0.8" />
        {/* Labels */}
        <text x="20" y="108" fontSize="9" fill="#64748b">
          {minTemp}°
        </text>
        <text x="170" y="108" fontSize="9" fill="#64748b">
          {maxTemp}°
        </text>
      </svg>
      <div className="text-center -mt-2">
        <span
          className="text-3xl font-black"
          style={{ color: needleColor }}
        >
          {value.toFixed(1)}°C
        </span>
      </div>
    </div>
  );
}

// SVG arc helper
function polarToCartesian(cx: number, cy: number, r: number, deg: number) {
  const rad = (Math.PI / 180) * deg;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
function describeArc(
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number
) {
  const start = polarToCartesian(cx, cy, r, endDeg);
  const end = polarToCartesian(cx, cy, r, startDeg);
  const largeArc = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}

// ── Anomaly Injection Buttons ─────────────────────────────────────────────────
const ANOMALY_BUTTONS = [
  { type: "temperature_spike", label: "Thermal Spike", icon: Flame, color: "rose" },
  { type: "bearing_wear", label: "Bearing Wear", icon: Settings, color: "amber" },
  { type: "fan_blockage", label: "Fan Blockage", icon: Wind, color: "orange" },
  { type: "sensor_drift", label: "Sensor Drift", icon: Radio, color: "blue" },
  { type: "power_surge", label: "Power Surge", icon: BatteryWarning, color: "fuchsia" },
];

const colorMap: Record<string, string> = {
  rose: "bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border-rose-500/30",
  amber: "bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border-amber-500/30",
  orange: "bg-orange-500/10 hover:bg-orange-500/20 text-orange-400 border-orange-500/30",
  blue: "bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border-blue-500/30",
  fuchsia: "bg-fuchsia-500/10 hover:bg-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/30",
};

// ── Main Dashboard Page ───────────────────────────────────────────────────────
export default function DashboardPage() {
  const { latest, history, injectAnomaly, resetSystem, systemStatus } =
    useTelemetry();

  // Prepare chart data
  const chartData = history.map((t) => ({
    ...t,
    time: timeLabel(t.timestamp),
  }));

  const thresholds = latest?.thresholds || { idle: 40, safe_max: 80, critical: 95 };

  return (
    <div className="p-6 space-y-6 relative overflow-hidden min-h-screen">
      {/* Background Aurora Glow */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-teal-500/8 blur-[180px] rounded-full pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-600/8 blur-[180px] rounded-full pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 flex items-center justify-between border-b border-slate-800/60 pb-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Live Monitoring</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Real-time sensor telemetry and AI response visualization
          </p>
        </div>
        {latest?.active_anomaly && (
          <div className="flex items-center gap-2 bg-rose-500/10 border border-rose-500/30 px-4 py-2 rounded-full animate-pulse">
            <ShieldAlert className="w-4 h-4 text-rose-400" />
            <span className="text-sm font-medium text-rose-400">
              Active: {latest.active_anomaly.replace("_", " ")} ({latest.anomaly_ticks_remaining}s)
            </span>
          </div>
        )}
        {latest?.metadata?.concept_drift_detected && (
          <div className="flex items-center gap-2 bg-amber-500/10 border border-amber-500/30 px-4 py-2 rounded-full animate-bounce">
            <Radio className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-400">
              Concept Drift Detected
            </span>
          </div>
        )}
      </header>

      {/* Top Row: Gauge + Action + Route */}
      <div className="relative z-10 grid grid-cols-12 gap-5">
        {/* Temperature Gauge */}
        <div className="col-span-12 md:col-span-4 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5" /> Temperature Gauge
          </h2>
          <TemperatureGauge
            value={latest?.current_temp ?? thresholds.idle}
            idle={thresholds.idle}
            safeMax={thresholds.safe_max}
            critical={thresholds.critical}
          />
          <div className="flex justify-between text-[10px] text-slate-500 mt-3 px-2">
            <span>
              Idle: <span className="text-emerald-400">{thresholds.idle}°C</span>
            </span>
            <span>
              Safe: <span className="text-amber-400">{thresholds.safe_max}°C</span>
            </span>
            <span>
              Critical: <span className="text-rose-400">{thresholds.critical}°C</span>
            </span>
          </div>
        </div>

        {/* RL Action Card */}
        <div className="col-span-6 md:col-span-4">
          <div
            className={`h-full rounded-2xl p-5 border shadow-xl transition-colors duration-500 ${
              latest ? getActionStyle(latest.action) : "bg-slate-900/50 border-slate-800"
            }`}
          >
            <div className="text-xs font-semibold uppercase tracking-wider opacity-70 mb-2 flex items-center gap-2">
              <Fan className="w-4 h-4" /> RL Agent Action
            </div>
            <div className="text-3xl font-black uppercase tracking-tight">
              {latest?.action || "STANDBY"}
            </div>
            <p className="text-xs opacity-60 mt-3">
              Urgency: {(latest?.urgency ?? 0).toFixed(3)}
            </p>
          </div>
        </div>

        {/* Routing Card */}
        <div className="col-span-6 md:col-span-4">
          <div className="h-full bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl relative overflow-hidden">
            {latest?.route === "edge" && (
              <div className="absolute inset-0 bg-teal-500/5 animate-pulse pointer-events-none" />
            )}
            {latest?.route === "cloud" && (
              <div className="absolute inset-0 bg-blue-500/5 animate-pulse pointer-events-none" />
            )}
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
              <Wifi className="w-4 h-4" /> Inference Routing
            </div>
            <div
              className={`text-3xl font-black uppercase tracking-tight ${
                latest?.route === "edge"
                  ? "text-teal-400"
                  : latest?.route === "cloud"
                  ? "text-blue-400"
                  : "text-purple-400"
              }`}
            >
              {latest?.route || "NONE"}
            </div>
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-slate-500">
                Source: <span className="text-slate-300">{latest?.llm_source || "—"}</span>
              </span>
              <span className="text-sm font-mono text-slate-500">
                {latest ? `${latest.latency_ms.toFixed(1)} ms` : "—"}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Middle Row: Multi-model Graph + Temp/LSTM Graph */}
      <div className="relative z-10 grid grid-cols-12 gap-5">
        {/* Multi-Model Anomaly Score Graph */}
        <div className="col-span-12 lg:col-span-7 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Multi-Model Anomaly Scores (Live)
          </h2>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="gradIF" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradLSTM" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradAE" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradVib" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="time" stroke="#475569" fontSize={10} tickMargin={6} />
                <YAxis stroke="#475569" fontSize={10} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b", borderRadius: "8px", fontSize: "12px" }}
                  itemStyle={{ color: "#e2e8f0" }}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: "11px" }} />
                <Area type="monotone" dataKey="if_score" name="Isolation Forest" stroke="#06b6d4" fill="url(#gradIF)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="lstm_score" name="LSTM Deviation" stroke="#14b8a6" fill="url(#gradLSTM)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="ae_score" name="Autoencoder" stroke="#f59e0b" fill="url(#gradAE)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="vib_score" name="Vibration" stroke="#a855f7" fill="url(#gradVib)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="context_score" name="Fused Score" stroke="#ffffff" strokeWidth={2} dot={false} isAnimationActive={false} strokeDasharray="4 4" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Fused Context + Urgency Gauges */}
        <div className="col-span-12 lg:col-span-5 space-y-5">
          {/* Context Score */}
          <div className="bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Fused Context Score
            </h2>
            <div className="flex items-center gap-6">
              <div className="relative flex items-center justify-center w-24 h-24 rounded-full border-4 border-slate-800 bg-slate-900/80">
                <div className="text-2xl font-black text-white">
                  {(latest?.context_score ?? 0).toFixed(2)}
                </div>
                <div
                  className={`absolute inset-0 rounded-full border-4 transition-all duration-500 ${
                    latest && latest.context_score > 0.6
                      ? "border-rose-500 animate-pulse"
                      : "border-transparent"
                  }`}
                />
              </div>
              <div className="flex-1 space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">System Urgency</span>
                    <span className="text-slate-200">{(latest?.urgency ?? 0).toFixed(3)}</span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${(latest?.urgency ?? 0) * 100}%`,
                        background:
                          (latest?.urgency ?? 0) > 0.8
                            ? "#ef4444"
                            : (latest?.urgency ?? 0) > 0.5
                            ? "#f59e0b"
                            : "#6366f1",
                      }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Raw Anomaly (AE)</span>
                    <span className="text-slate-200">{(latest?.anomaly_score ?? 0).toFixed(3)}</span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full transition-all duration-500"
                      style={{ width: `${(latest?.anomaly_score ?? 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* LLM Alert */}
          <div className="bg-[#0b0f19] border border-slate-800/80 rounded-2xl p-5 shadow-xl flex flex-col h-[196px]">
            <h2 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-2 flex items-center gap-2">
              <Zap className="w-3.5 h-3.5" /> Phi-3 Context Engine
              {latest?.llm_source && (
                <span className="ml-auto text-[9px] bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-500/30">
                  {latest.llm_source}
                </span>
              )}
            </h2>
            <div className="flex-1 bg-black/40 rounded-lg p-3 overflow-y-auto border border-slate-800/50 font-mono text-xs leading-relaxed text-slate-300">
              {latest ? (
                <div className={latest.urgency > 0.6 ? "text-rose-200" : "text-emerald-200"}>
                  {latest.alert}
                </div>
              ) : (
                <span className="text-slate-600 animate-pulse">
                  Waiting for telemetry sequence...
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Row: Temp Chart + Injection Panel */}
      <div className="relative z-10 grid grid-cols-12 gap-5">
        {/* Temperature + LSTM Prediction */}
        <div className="col-span-12 lg:col-span-8 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Temperature vs LSTM Prediction
          </h2>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="time" stroke="#475569" fontSize={10} tickMargin={6} />
                <YAxis stroke="#475569" fontSize={10} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b", borderRadius: "8px", fontSize: "12px" }}
                  itemStyle={{ color: "#e2e8f0" }}
                />
                <Legend iconType="circle" wrapperStyle={{ fontSize: "11px" }} />
                <ReferenceLine
                  y={thresholds.safe_max}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  label={{ value: "Safe Max", fill: "#f59e0b", fontSize: 10, position: "right" }}
                />
                <ReferenceLine
                  y={thresholds.critical}
                  stroke="#ef4444"
                  strokeDasharray="4 4"
                  label={{ value: "Critical", fill: "#ef4444", fontSize: 10, position: "right" }}
                />
                <Line type="monotone" dataKey="current_temp" name="Actual Temp (°C)" stroke="#f43f5e" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="lstm_prediction" name="LSTM Predict (+10m)" stroke="#14b8a6" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Anomaly Injection Panel */}
        <div className="col-span-12 lg:col-span-4 bg-gradient-to-br from-indigo-950/40 to-slate-900/50 backdrop-blur-md border border-indigo-900/50 rounded-2xl p-5 shadow-xl">
          <h2 className="text-xs font-semibold text-indigo-300 uppercase tracking-wider mb-2 flex items-center gap-2">
            <ShieldAlert className="w-3.5 h-3.5" /> Instructor Mode
          </h2>
          <p className="text-[10px] text-slate-500 mb-4">
            Inject anomalies to observe live AI response and routing logic.
          </p>
          <div className="space-y-2.5">
            {ANOMALY_BUTTONS.map((btn) => (
              <button
                key={btn.type}
                onClick={() => injectAnomaly(btn.type)}
                className={`w-full flex items-center gap-2.5 border rounded-lg py-2 px-3 text-sm font-medium transition-colors ${
                  colorMap[btn.color]
                }`}
              >
                <btn.icon className="w-4 h-4" />
                {btn.label}
              </button>
            ))}
            <button
              onClick={resetSystem}
              className="w-full flex items-center gap-2.5 border border-slate-700 rounded-lg py-2 px-3 text-sm font-medium text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors mt-3"
            >
              <RotateCcw className="w-4 h-4" />
              Reset System
            </button>
          </div>
        </div>
      </div>

      {/* Fleet Registry (Federated Network) */}
      <div className="relative z-10 bg-slate-900/40 backdrop-blur-md border border-slate-800/60 rounded-2xl p-5 shadow-xl mt-6">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <Network className="w-4 h-4 text-indigo-400" /> Federated Fleet Registry
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(useTelemetry().registry).map(([nodeId, data]: [string, any]) => (
            <div key={nodeId} className="bg-slate-950/50 border border-slate-800/50 rounded-xl p-3 flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-slate-200">{nodeId}</div>
                <div className="text-[10px] text-slate-500 uppercase">{data.machine_type}</div>
              </div>
              <div className="text-right">
                <div className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                  data.status === "active" || data.status === "do-nothing"
                    ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/10"
                    : "text-rose-400 border-rose-500/20 bg-rose-500/10"
                }`}>
                  {data.status}
                </div>
                <div className="text-[9px] text-slate-600 mt-1">{data.last_updated}</div>
              </div>
            </div>
          ))}
          {Object.keys(useTelemetry().registry).length === 0 && (
            <div className="col-span-full py-8 text-center text-slate-600 text-sm italic">
              No other active nodes detected in the local network.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
