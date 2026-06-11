"use client";

import React from "react";
import {
  BookOpen,
  ArrowRight,
  Cpu,
  Activity,
  Wifi,
  ShieldAlert,
  Gauge,
  Radio,
  Layers,
  Brain,
  Filter,
  Network,
  Eye,
} from "lucide-react";
import { Card, CardTitle, PageHeader, Badge } from "@/components/ui";

const STAGES = [
  { icon: Activity, name: "Sensor", desc: "Temp, load, vibration, ambient captured at 1 Hz." },
  { icon: Filter, name: "Noise Filter", desc: "Isolation Forest rejects corrupt / outlier readings." },
  { icon: Brain, name: "ML Models", desc: "LSTM forecast + Autoencoder + Vibration + CNN + Audio." },
  { icon: Layers, name: "Fusion", desc: "Attention layer blends 6 modalities → one context score." },
  { icon: Eye, name: "Uncertainty", desc: "MC-Dropout + conformal bounds gauge confidence." },
  { icon: Network, name: "RL Router", desc: "Urgency decides edge vs cloud inference." },
  { icon: Cpu, name: "PPO Action", desc: "Agent picks fan+, throttle, alert or shutdown." },
  { icon: ShieldAlert, name: "LLM Alert", desc: "Phi-3 writes a plain-language operator alert." },
];

const MODELS = [
  { tag: "Heat Trend", name: "BiLSTM forecaster", what: "Predicts the next ~10 steps of temperature (a short forward trajectory). A big gap between now and the forecast means trouble is building." },
  { tag: "Pattern Match", name: "Autoencoder", what: "Learns what a healthy signal looks like. If live data no longer matches that shape, it flags it." },
  { tag: "Odd Reading", name: "Isolation Forest", what: "Catches out-of-range or corrupt sensor values. First line of defence against noise and spikes." },
  { tag: "Vibration", name: "Vibration detector", what: "Spots mechanical faults like bearing wear or imbalance from vibration." },
  { tag: "Heat Camera", name: "Vision CNN", what: "Looks for hot spots in thermal-camera images." },
  { tag: "Sound", name: "Audio model", what: "Listens for abnormal noises that signal failure." },
];

const PANELS = [
  { name: "Hero stats", reads: "Temperature, composite health, the AI's decision and time-to-limit at a glance." },
  { name: "KPI strip", reads: "Session uptime, peak temp, recent events, actions taken, average response time." },
  { name: "Temperature chart", reads: "Live °C vs the AI forecast, with SAFE and LIMIT reference lines." },
  { name: "Live Vitals", reads: "Real per-component readings (CPU/RAM or RPM/Bearing/etc.) for the active machine." },
  { name: "Anomaly Signals", reads: "Each model's suspicion level (0–1); the white line is the combined verdict." },
  { name: "Why this decision", reads: "How much each model influenced the decision; the main-reason line names the biggest factor." },
  { name: "Maintenance Plan", reads: "When to service the machine, with an urgency level and a recommended window." },
  { name: "Event Log", reads: "Recent warnings/faults with timestamps. Click Ack to acknowledge." },
  { name: "AI Assistant", reads: "A plain-language alert written by the on-device Phi-3 language model." },
];

const GLOSSARY = [
  { term: "Anomaly", plain: "Something unusual — a reading or pattern that doesn't fit normal operation." },
  { term: "Urgency / Confidence", plain: "How sure and how alarmed the AI is, from 0 (calm) to 1 (act now)." },
  { term: "RUL", plain: "Remaining Useful Life — estimated time before a limit is reached." },
  { term: "Fusion", plain: "Combining all six AI opinions into one overall verdict." },
  { term: "XAI", plain: "Explainable AI — showing why a decision was made, not just the result." },
  { term: "Concept drift", plain: "The 'normal' range slowly changing over time (e.g. seasonal warming)." },
  { term: "Edge vs Cloud", plain: "Edge = computed on the machine itself; Cloud = computed on a remote server." },
  { term: "Conformal interval", plain: "A statistical 'best–worst case' range around a prediction." },
];

const COLORS = [
  { c: "#34d399", name: "Green", mean: "Safe / healthy / no action needed." },
  { c: "#fbbf24", name: "Amber", mean: "Caution — approaching a limit." },
  { c: "#f87171", name: "Red", mean: "Critical — over the limit or fault." },
  { c: "#22d3ee", name: "Cyan", mean: "AI forecast / active signal / highlight." },
];

const ANALYTICS = [
  { name: "Composite Health ring", reads: "0–100 score from thermal, mechanical and longevity sub-scores." },
  { name: "RUL countdown", reads: "Estimated days/hours/min until the machine reaches its safe limit." },
  { name: "Feature attribution", reads: "Gradient-based XAI: which input sensor drove the model output." },
  { name: "Uncertainty band", reads: "Shaded 95% interval around the forecast from MC-Dropout / conformal prediction." },
  { name: "CUSUM / EWMA chart", reads: "Statistical drift detection — catches slow shifts too gradual for the anomaly models." },
];

const ROLES = [
  { role: "Viewer", tone: "info", can: "View dashboards, download reports." },
  { role: "Operator", tone: "warn", can: "+ Inject faults, switch machine, create share links, toggle live/demo." },
  { role: "Admin", tone: "good", can: "+ Manage users and revoke any share link." },
];

export default function LearnPage() {
  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      <PageHeader icon={BookOpen} title="How It Works" subtitle="A guided tour of the MHARS pipeline and every panel" accent="#818cf8" />

      <Card accent>
        <CardTitle icon={Activity}>What is MHARS</CardTitle>
        <p className="text-[14px] text-slate-300 leading-relaxed">
          MHARS (Multi-modal Hybrid Adaptive Response System) is a digital twin for thermal-critical machines.
          It fuses six AI models to detect anomalies, forecast failures, decide a protective action, and explain
          itself — all in real time. Below is the exact path a single sensor reading takes every second.
        </p>
      </Card>

      <Card>
        <CardTitle icon={Network}>The Pipeline · 1 Hz Loop</CardTitle>
        <div className="flex flex-wrap items-stretch gap-2">
          {STAGES.map((s, i) => (
            <React.Fragment key={s.name}>
              <div className="flex-1 min-w-[150px] rounded-xl border border-white/[0.06] bg-white/[0.02] p-3.5">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="metric text-indigo-400 text-xs">{String(i + 1).padStart(2, "0")}</span>
                  <s.icon className="w-4 h-4 text-indigo-300" />
                  <span className="text-[13px] font-medium text-slate-100">{s.name}</span>
                </div>
                <p className="text-[11px] text-slate-500 leading-snug">{s.desc}</p>
              </div>
              {i < STAGES.length - 1 && (
                <div className="hidden xl:flex items-center"><ArrowRight className="w-4 h-4 text-slate-700" /></div>
              )}
            </React.Fragment>
          ))}
        </div>
      </Card>

      <Card>
        <CardTitle icon={Brain}>The Six Models</CardTitle>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {MODELS.map((m) => (
            <div key={m.tag} className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
              <div className="eyebrow text-cyan-400/90">{m.tag}</div>
              <div className="text-sm text-slate-100 font-medium mt-0.5 mb-1.5">{m.name}</div>
              <p className="text-xs text-slate-400 leading-relaxed">{m.what}</p>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-7">
          <CardTitle icon={Gauge}>Reading the Overview</CardTitle>
          <div className="space-y-2">
            {PANELS.map((p) => (
              <div key={p.name} className="flex gap-3 border-b border-white/[0.05] pb-2 last:border-0">
                <div className="text-[12px] text-cyan-300/90 w-40 shrink-0">{p.name}</div>
                <div className="text-xs text-slate-400 leading-snug">{p.reads}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-5">
          <CardTitle icon={Layers}>Reading the Analytics Page</CardTitle>
          <div className="space-y-2">
            {ANALYTICS.map((p) => (
              <div key={p.name} className="flex gap-3 border-b border-white/[0.05] pb-2 last:border-0">
                <div className="text-[12px] text-cyan-300/90 w-32 shrink-0">{p.name}</div>
                <div className="text-xs text-slate-400 leading-snug">{p.reads}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-6">
          <CardTitle icon={Radio}>Demo vs Live</CardTitle>
          <div className="space-y-3">
            <div className="rounded-xl border border-blue-500/25 bg-blue-500/5 p-3.5">
              <div className="text-[12px] text-blue-300 font-medium mb-1">Demo · Simulation</div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Runs a simulated thermal environment. Use the Test Panel to inject faults
                (heat spike, bearing wear, fan blockage, sensor drift, power surge) and watch the AI react.
              </p>
            </div>
            <div className="rounded-xl border border-rose-500/25 bg-rose-500/5 p-3.5">
              <div className="text-[12px] text-rose-300 font-medium mb-1">Live · Real Hardware</div>
              <p className="text-xs text-slate-400 leading-relaxed">
                Reads your actual machine&apos;s CPU temperature / load via system sensors and runs the
                same pipeline on real telemetry. Toggle from the sidebar (operator+).
              </p>
            </div>
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-6">
          <CardTitle icon={ShieldAlert}>Access Roles · RBAC</CardTitle>
          <div className="space-y-2">
            {ROLES.map((r) => (
              <div key={r.role} className="flex items-start gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                <Badge tone={r.tone}>{r.role}</Badge>
                <span className="text-xs text-slate-400 leading-snug flex-1">{r.can}</span>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-600 mt-3">Demo logins: admin/admin123 · operator/oper123 · viewer/view123</p>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-8">
          <CardTitle icon={BookOpen}>Glossary · Plain English</CardTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
            {GLOSSARY.map((g) => (
              <div key={g.term} className="flex gap-2 border-b border-white/[0.05] pb-2">
                <span className="text-[12px] text-cyan-300/90 w-32 shrink-0">{g.term}</span>
                <span className="text-xs text-slate-400 leading-snug">{g.plain}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-4">
          <CardTitle icon={Eye}>Color Code</CardTitle>
          <div className="space-y-2.5">
            {COLORS.map((c) => (
              <div key={c.name} className="flex items-center gap-3">
                <span className="inline-block w-4 h-4 rounded shrink-0" style={{ background: c.c }} />
                <span className="text-[12px] text-slate-200 w-14">{c.name}</span>
                <span className="text-xs text-slate-400 leading-snug flex-1">{c.mean}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card>
        <div className="flex items-center gap-2 text-[12px] text-slate-500">
          <Wifi className="w-3.5 h-3.5 text-teal-400" />
          Telemetry streams over WebSocket at 1 Hz. Every number you see is a live model output — nothing is mocked.
        </div>
      </Card>
    </div>
  );
}
