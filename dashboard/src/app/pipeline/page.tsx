"use client";

import React from "react";
import { useTelemetry } from "@/components/TelemetryProvider";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  Legend,
} from "recharts";
import { GitBranch, Eye } from "lucide-react";

// ── Helper ────────────────────────────────────────────────────────────────────
function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
  });
}

// ── Pipeline Flow Diagram ─────────────────────────────────────────────────────
function PipelineFlow({
  ifScore,
  lstmScore,
  aeScore,
  vibScore,
  contextScore,
  urgency,
  action,
  route,
}: {
  ifScore: number;
  lstmScore: number;
  aeScore: number;
  vibScore: number;
  contextScore: number;
  urgency: number;
  action: string;
  route: string;
}) {
  const nodeColor = (score: number) =>
    score > 0.6
      ? "border-rose-500 shadow-[0_0_12px_rgba(239,68,68,0.4)]"
      : score > 0.3
      ? "border-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.3)]"
      : "border-emerald-500/50";

  const nodeGlow = (score: number) =>
    score > 0.6
      ? "bg-rose-500/10"
      : score > 0.3
      ? "bg-amber-500/10"
      : "bg-emerald-500/5";

  const nodes = [
    { label: "Isolation Forest", score: ifScore, sub: "Noise Filter" },
    { label: "LSTM Predictor", score: lstmScore, sub: "+10min Forecast" },
    { label: "Autoencoder", score: aeScore, sub: "Reconstruction" },
    { label: "Vibration Model", score: vibScore, sub: "Bearing Health" },
  ];

  return (
    <div className="space-y-6">
      {/* Input */}
      <div className="flex items-center justify-center">
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl px-6 py-3 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">Input</div>
          <div className="text-sm font-bold text-slate-200">Sensor Telemetry</div>
        </div>
      </div>

      {/* Arrow */}
      <div className="flex justify-center">
        <div className="w-px h-6 bg-slate-700" />
      </div>

      {/* 4 Models */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {nodes.map((node) => (
          <div
            key={node.label}
            className={`rounded-xl border-2 p-4 text-center transition-all duration-500 ${nodeColor(
              node.score
            )} ${nodeGlow(node.score)}`}
          >
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
              {node.sub}
            </div>
            <div className="text-xs font-bold text-slate-200 mb-2">{node.label}</div>
            <div
              className={`text-xl font-black ${
                node.score > 0.6
                  ? "text-rose-400"
                  : node.score > 0.3
                  ? "text-amber-400"
                  : "text-emerald-400"
              }`}
            >
              {node.score.toFixed(3)}
            </div>
          </div>
        ))}
      </div>

      {/* Arrow */}
      <div className="flex justify-center">
        <div className="w-px h-6 bg-slate-700" />
      </div>

      {/* Fusion */}
      <div className="flex items-center justify-center">
        <div
          className={`rounded-xl border-2 p-4 text-center w-64 transition-all duration-500 ${nodeColor(
            contextScore
          )} ${nodeGlow(contextScore)}`}
        >
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Attention-Weighted
          </div>
          <div className="text-xs font-bold text-slate-200 mb-2">Fusion Layer</div>
          <div className="flex items-center justify-center gap-4">
            <div>
              <div className="text-[10px] text-slate-500">Context</div>
              <div className="text-lg font-black text-white">{contextScore.toFixed(3)}</div>
            </div>
            <div className="w-px h-8 bg-slate-700" />
            <div>
              <div className="text-[10px] text-slate-500">Urgency</div>
              <div className="text-lg font-black text-white">{urgency.toFixed(3)}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Arrow */}
      <div className="flex justify-center">
        <div className="w-px h-6 bg-slate-700" />
      </div>

      {/* PPO + Router */}
      <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto">
        <div className="rounded-xl border-2 border-purple-500/50 bg-purple-500/5 p-4 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Reinforcement Learning
          </div>
          <div className="text-xs font-bold text-slate-200 mb-2">PPO Agent</div>
          <div className="text-lg font-black text-purple-400 uppercase">{action}</div>
        </div>
        <div
          className={`rounded-xl border-2 p-4 text-center ${
            route === "edge"
              ? "border-teal-500/50 bg-teal-500/5"
              : route === "cloud"
              ? "border-blue-500/50 bg-blue-500/5"
              : "border-indigo-500/50 bg-indigo-500/5"
          }`}
        >
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
            Inference Path
          </div>
          <div className="text-xs font-bold text-slate-200 mb-2">Router</div>
          <div
            className={`text-lg font-black uppercase ${
              route === "edge"
                ? "text-teal-400"
                : route === "cloud"
                ? "text-blue-400"
                : "text-indigo-400"
            }`}
          >
            {route}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── PPO Observation Vector ────────────────────────────────────────────────────
const OBS_LABELS = [
  "Temp (norm)",
  "Predict (norm)",
  "AE Score",
  "Machine Type",
  "Steps Since Act",
  "Urgency",
];

function ObsVector({ obs }: { obs: number[] }) {
  return (
    <div className="grid grid-cols-3 gap-3">
      {obs.map((val, idx) => (
        <div
          key={idx}
          className="bg-slate-800/60 border border-slate-700/60 rounded-lg p-3 text-center"
        >
          <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1">
            {OBS_LABELS[idx] || `dim[${idx}]`}
          </div>
          <div className="text-lg font-black text-slate-200 font-mono">
            {val.toFixed(3)}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function PipelinePage() {
  const { latest, history } = useTelemetry();

  const chartData = history.map((t) => ({
    ...t,
    time: timeLabel(t.timestamp),
  }));

  // Per-model bar data
  const barData = [
    { name: "Isolation\nForest", score: latest?.if_score ?? 0, color: "#06b6d4" },
    { name: "LSTM", score: latest?.lstm_score ?? 0, color: "#14b8a6" },
    { name: "Autoencoder", score: latest?.ae_score ?? 0, color: "#f59e0b" },
    { name: "Vibration", score: latest?.vib_score ?? 0, color: "#a855f7" },
  ];

  return (
    <div className="p-6 space-y-6 relative overflow-hidden min-h-screen">
      {/* Background */}
      <div className="absolute top-[-15%] right-[-5%] w-[40%] h-[40%] bg-indigo-600/8 blur-[150px] rounded-full pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 border-b border-slate-800/60 pb-4">
        <h1 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
          <GitBranch className="w-6 h-6 text-indigo-400" />
          AI Pipeline Deep Dive
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Interactive visualization of every AI model in the MHARS decision chain
        </p>
      </header>

      {/* Pipeline Flow */}
      <div className="relative z-10 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-6 shadow-xl">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-6">
          Live Pipeline State
        </h2>
        <PipelineFlow
          ifScore={latest?.if_score ?? 0}
          lstmScore={latest?.lstm_score ?? 0}
          aeScore={latest?.ae_score ?? 0}
          vibScore={latest?.vib_score ?? 0}
          contextScore={latest?.context_score ?? 0}
          urgency={latest?.urgency ?? 0}
          action={latest?.action ?? "standby"}
          route={latest?.route ?? "none"}
        />
      </div>

      {/* Charts Row */}
      <div className="relative z-10 grid grid-cols-12 gap-5">
        {/* Per-Model Bar Chart */}
        <div className="col-span-12 lg:col-span-5 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Per-Model Scores (Current)
          </h2>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="name" stroke="#475569" fontSize={10} tickMargin={6} />
                <YAxis stroke="#475569" fontSize={10} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    borderColor: "#1e293b",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                />
                <Bar dataKey="score" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency History */}
        <div className="col-span-12 lg:col-span-7 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Inference Latency Over Time (ms)
          </h2>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                <XAxis dataKey="time" stroke="#475569" fontSize={10} tickMargin={6} />
                <YAxis stroke="#475569" fontSize={10} />
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
                <Line
                  type="monotone"
                  dataKey="latency_ms"
                  name="Latency (ms)"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* PPO Observation Vector */}
      <div className="relative z-10 bg-slate-900/50 backdrop-blur-md border border-slate-800/80 rounded-2xl p-5 shadow-xl">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <Eye className="w-3.5 h-3.5" /> PPO Agent Observation Vector (What the RL Agent Sees)
        </h2>
        <ObsVector obs={latest?.raw_obs ?? [0, 0, 0, 0, 0, 0]} />
      </div>
    </div>
  );
}
