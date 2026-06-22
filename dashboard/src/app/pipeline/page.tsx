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
import { Card, CardTitle, PageHeader, CHART, tooltipStyle, tooltipLabelStyle } from "@/components/ui";

function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour12: false, minute: "2-digit", second: "2-digit" });
}

function scoreTone(score: number) {
  if (score > 0.6) return { border: "border-rose-500/60", glow: "bg-rose-500/10", text: "text-rose-700" };
  if (score > 0.3) return { border: "border-amber-500/60", glow: "bg-amber-500/10", text: "text-amber-700" };
  return { border: "border-emerald-500/40", glow: "bg-emerald-500/5", text: "text-emerald-700" };
}

function Node({ label, sub, score }: { label: string; sub: string; score: number }) {
  const t = scoreTone(score);
  return (
    <div className={`rounded-xl border ${t.border} ${t.glow} p-4 text-center transition-all duration-500`}>
      <div className="eyebrow mb-1">{sub}</div>
      <div className="text-[13px] font-semibold text-[var(--text)] mb-2">{label}</div>
      <div className={`metric text-xl ${t.text}`}>{score.toFixed(3)}</div>
    </div>
  );
}

function Connector() {
  return <div className="flex justify-center"><div className="w-px h-6 bg-gradient-to-b from-[var(--border-strong)] to-transparent" /></div>;
}

function PipelineFlow({ ifScore, lstmScore, aeScore, vibScore, contextScore, urgency, action, route }: {
  ifScore: number; lstmScore: number; aeScore: number; vibScore: number;
  contextScore: number; urgency: number; action: string; route: string;
}) {
  const ct = scoreTone(contextScore);
  const routeColor = route === "edge" ? "border-cyan-500/50 bg-cyan-500/5 text-sky-700"
    : route === "cloud" ? "border-blue-500/50 bg-blue-500/5 text-blue-700"
    : "border-indigo-500/50 bg-indigo-500/5 text-indigo-600";
  return (
    <div className="space-y-5">
      <div className="flex justify-center">
        <div className="rounded-xl bg-[var(--surface-3)] border border-[var(--border)] px-6 py-3 text-center">
          <div className="eyebrow">Input</div>
          <div className="text-sm font-semibold text-[var(--text)] mt-0.5">Sensor Telemetry</div>
        </div>
      </div>
      <Connector />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <Node label="Isolation Forest" sub="Noise Filter" score={ifScore} />
        <Node label="LSTM Predictor" sub="+10-step Forecast" score={lstmScore} />
        <Node label="Autoencoder" sub="Reconstruction" score={aeScore} />
        <Node label="Vibration Model" sub="Bearing Health" score={vibScore} />
      </div>
      <Connector />
      <div className="flex justify-center">
        <div className={`rounded-xl border ${ct.border} ${ct.glow} p-4 text-center w-72 transition-all duration-500`}>
          <div className="eyebrow mb-1">Attention-Weighted</div>
          <div className="text-[13px] font-semibold text-[var(--text)] mb-2">Fusion Layer</div>
          <div className="flex items-center justify-center gap-5">
            <div><div className="text-[10px] text-[var(--text-dim)]">Context</div><div className="metric text-lg text-[var(--text)]">{contextScore.toFixed(3)}</div></div>
            <div className="w-px h-8 bg-[var(--surface-3)]" />
            <div><div className="text-[10px] text-[var(--text-dim)]">Urgency</div><div className="metric text-lg text-[var(--text)]">{urgency.toFixed(3)}</div></div>
          </div>
        </div>
      </div>
      <Connector />
      <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto">
        <div className="rounded-xl border border-indigo-500/50 bg-indigo-500/5 p-4 text-center">
          <div className="eyebrow mb-1">Reinforcement Learning</div>
          <div className="text-[13px] font-semibold text-[var(--text)] mb-2">PPO Agent</div>
          <div className="metric text-lg text-indigo-600 uppercase">{action}</div>
        </div>
        <div className={`rounded-xl border ${routeColor} p-4 text-center`}>
          <div className="eyebrow mb-1">Inference Path</div>
          <div className="text-[13px] font-semibold text-[var(--text)] mb-2">Router</div>
          <div className="metric text-lg uppercase">{route}</div>
        </div>
      </div>
    </div>
  );
}

const OBS_LABELS = ["Temp (norm)", "Predict (norm)", "AE Score", "Machine Type", "Steps Since Act", "Urgency"];

export default function PipelinePage() {
  const { latest, history } = useTelemetry();
  const chartData = history.map((t) => ({ ...t, time: timeLabel(t.timestamp) }));
  const barData = [
    { name: "Isolation\nForest", score: latest?.if_score ?? 0, color: CHART.cyan },
    { name: "LSTM", score: latest?.lstm_score ?? 0, color: CHART.teal },
    { name: "Autoencoder", score: latest?.ae_score ?? 0, color: CHART.amber },
    { name: "Vibration", score: latest?.vib_score ?? 0, color: CHART.indigo },
  ];
  const obs = latest?.raw_obs ?? [0, 0, 0, 0, 0, 0];

  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      <PageHeader icon={GitBranch} title="AI Pipeline Deep Dive" subtitle="Every model in the MHARS decision chain, live" accent="#5e6ad2" />

      <Card>
        <CardTitle icon={GitBranch}>Live Pipeline State</CardTitle>
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
      </Card>

      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-5">
          <CardTitle>Per-Model Scores (Current)</CardTitle>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="name" stroke={CHART.axis} fontSize={10} tickMargin={6} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART.axis} fontSize={10} domain={[0, 1]} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                <Bar dataKey="score" radius={[6, 6, 0, 0]}>
                  {barData.map((e, i) => (<Cell key={i} fill={e.color} />))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-7">
          <CardTitle>Inference Latency Over Time (ms)</CardTitle>
          <div className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="time" stroke={CHART.axis} fontSize={10} tickMargin={6} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={{ color: "#e2e8f0" }} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="latency_ms" name="Latency (ms)" stroke={CHART.indigo} strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card>
        <CardTitle icon={Eye}>PPO Agent Observation Vector (What the RL Agent Sees)</CardTitle>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {obs.map((val, idx) => (
            <div key={idx} className="rounded-xl bg-[var(--surface-3)] border border-[var(--border)] p-3 text-center">
              <div className="eyebrow mb-1">{OBS_LABELS[idx] || `dim[${idx}]`}</div>
              <div className="metric text-lg text-[var(--text)]">{val.toFixed(3)}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
