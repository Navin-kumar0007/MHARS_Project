"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTelemetry } from "@/components/TelemetryProvider";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Thermometer,
  Cpu,
  Wifi,
  Timer,
  ShieldAlert,
  Activity,
  FileDown,
  Flame,
  Settings,
  Wind,
  Radio,
  BatteryWarning,
  RotateCcw,
  HeartPulse,
  Clock,
  TrendingUp,
  Bell,
  Wrench,
  Zap,
  Check,
  Gauge,
  Sparkles,
} from "lucide-react";
import {
  Card,
  CardTitle,
  StatCard,
  Badge,
  PageHeader,
  Progress,
  Awaiting,
  CHART,
  tooltipStyle,
  tooltipLabelStyle,
  healthColor,
  healthLabel,
} from "@/components/ui";

// ── Helpers ─────────────────────────────────────────────────────────────────
function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour12: false, minute: "2-digit", second: "2-digit" });
}

function decisionInfo(action: string | undefined) {
  switch (action) {
    case "do-nothing":
      return { text: "All Clear", tone: "good", color: "#34d399", tip: "No action needed — machine is healthy." };
    case "fan+":
    case "increase-fan":
      return { text: "Cooling Up", tone: "warn", color: "#fbbf24", tip: "AI increased cooling to bring temperature down." };
    case "throttle":
      return { text: "Throttling", tone: "warn", color: "#fb923c", tip: "AI reduced workload to limit heat." };
    case "alert":
      return { text: "Warning", tone: "warn", color: "#fbbf24", tip: "AI raised a warning — operator attention advised." };
    case "shutdown":
    case "emergency-shutdown":
      return { text: "Shutdown", tone: "bad", color: "#ef4444", tip: "AI triggered an emergency stop to prevent damage." };
    default:
      return { text: "Standby", tone: "neutral", color: "#5d6b82", tip: "Waiting for telemetry." };
  }
}

function routeInfo(route: string | undefined) {
  if (route === "edge") return { color: "#22d3ee", tip: "Handled locally on the machine — fastest, used when urgent." };
  if (route === "cloud") return { color: "#60a5fa", tip: "Deferred to the cloud — used when there is time." };
  if (route === "both") return { color: "#818cf8", tip: "Run in both places at once for safety." };
  return { color: "#5d6b82", tip: "Where the AI runs its decision." };
}

const ANOMALY_BUTTONS = [
  { type: "temperature_spike", label: "Heat Spike", icon: Flame, color: "#ef4444", tip: "Sudden +8°C overload." },
  { type: "bearing_wear", label: "Bearing Wear", icon: Settings, color: "#fbbf24", tip: "Gradual friction heat over 10s." },
  { type: "fan_blockage", label: "Fan Blockage", icon: Wind, color: "#fb923c", tip: "Cooling failure for 15s." },
  { type: "sensor_drift", label: "Sensor Drift", icon: Radio, color: "#22d3ee", tip: "Faulty sensor adds ±3°C noise." },
  { type: "power_surge", label: "Power Surge", icon: BatteryWarning, color: "#e879f9", tip: "Electrical fault, +12°C spike." },
];

// Only the four trained modalities with real signal. CNN (Heat Camera) and
// Audio are gated out until trained on real data (P1.3), so they are not shown.
const XAI_ROWS: { key: string; label: string; meaning: string; model: string }[] = [
  { key: "trend_forecast", label: "Heat Trend", meaning: "Is temperature rising?", model: "LSTM forecaster" },
  { key: "pattern_check", label: "Pattern Match", meaning: "Does the signal look normal?", model: "Autoencoder" },
  { key: "outlier_scan", label: "Odd Reading", meaning: "Any out-of-range value?", model: "Isolation Forest" },
  { key: "vibration", label: "Vibration", meaning: "Shaking / bearing wear?", model: "Vibration model" },
];

function statusColor(s: string) {
  return s === "critical" ? "#f87171" : s === "warning" ? "#fbbf24" : "#34d399";
}

function fmtUptime(sec: number) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  return `${h ? h + "h " : ""}${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
}

function KpiTile({ icon: Icon, label, value, tip }: { icon: React.ElementType; label: string; value: string; tip: string }) {
  return (
    <Card hover className="!p-3.5 flex items-center gap-3" title={tip}>
      <div className="grid place-items-center w-9 h-9 rounded-lg bg-white/[0.04] shrink-0">
        <Icon className="w-4 h-4 text-slate-400" />
      </div>
      <div className="min-w-0">
        <div className="eyebrow leading-tight">{label}</div>
        <div className="metric text-sm text-slate-100 leading-tight mt-0.5">{value}</div>
      </div>
    </Card>
  );
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const { latest, history, injectAnomaly, resetSystem, downloadReport, can, liveMode, isConnected, systemStatus } = useTelemetry();
  const degraded = systemStatus?.models_degraded ?? [];
  const [clock, setClock] = useState("--:--:--");
  const [uptime, setUptime] = useState(0);
  const [acked, setAcked] = useState<Set<number>>(new Set());
  const startRef = useRef(Date.now());

  useEffect(() => {
    const id = setInterval(() => {
      setClock(new Date().toLocaleTimeString([], { hour12: false }));
      setUptime((Date.now() - startRef.current) / 1000);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const meta = latest?.metadata || {};
  const th = latest?.thresholds || { idle: 40, safe_max: 80, critical: 95 };
  const temp = latest?.current_temp;
  const tColor =
    temp != null ? (temp >= th.critical ? "#f87171" : temp >= th.safe_max ? "#fbbf24" : "#34d399") : "#5d6b82";
  const decision = decisionInfo(latest?.action);
  const route = routeInfo(latest?.route);
  const sh = latest?.system_health;
  const drift = !!meta.concept_drift_detected;

  // Canonical composite health (same number the sidebar + analytics use)
  const health = (meta.health_score ?? sh?.overall_score) as number | undefined;
  const hColor = healthColor(health);

  const concerned = latest?.action !== "do-nothing" || (latest?.urgency ?? 0) >= 0.7;
  const fault = concerned && meta.fault_type && meta.fault_type !== "Normal Operations" ? meta.fault_type : null;
  const faultConf = meta.fault_confidence as number | undefined;
  const maint = meta.maintenance_plan as
    | { status: string; urgency_level: string; action: string; window_start: string | null; window_end: string | null }
    | undefined;

  // Time-to-limit (RUL)
  const rul = meta.rul_minutes as number | null | undefined;
  let rulText = "Stable";
  let rulColor = "#34d399";
  if (rul != null && rul > 0) {
    if (rul > 60) { rulText = ">1h"; rulColor = "#34d399"; }
    else if (rul > 15) { rulText = `${Math.round(rul)}m`; rulColor = "#fbbf24"; }
    else { rulText = `${Math.round(rul)}m`; rulColor = "#f87171"; }
  }

  // XAI contributions
  let contrib = (meta.contributions || {}) as Record<string, number>;
  if (!meta.contributions && latest) {
    const raw: Record<string, number> = {
      trend_forecast: latest.lstm_score || 0,
      pattern_check: latest.ae_score || 0,
      outlier_scan: latest.if_score || 0,
      vibration: latest.vib_score || 0,
      cnn_hotspot: 0,
      audio: 0,
    };
    const total = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
    contrib = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, Math.round((v / total) * 100)]));
  }
  const topKey = meta.top_contributor as string | undefined;
  const topRow = XAI_ROWS.find((r) => r.key === topKey);

  const chartData = history.map((t) => ({ ...t, time: timeLabel(t.timestamp), detector: t.metadata?.anomaly_probability ?? 0 }));
  const tempSpark = history.slice(-30).map((t) => t.current_temp);
  // Detector: the sim-trained classifier is unreliable on real-CPU (LIVE) data,
  // so it's gated there — UNLESS the zero-shot foundation backbone is active,
  // which generalises to real hardware (R1). Then the detector is shown in Live.
  const foundationOn = meta.forecaster_backend === "foundation";
  const detectorGated = liveMode && !foundationOn;
  const detectorP = detectorGated ? undefined : (meta.anomaly_probability as number | undefined);

  // P1.5/P2.1: forward multi-horizon p50 trajectory + p10–p90 uncertainty band.
  const fcTraj = (meta.forecast_trajectory_c as number[] | null) || null;
  const fcBand = (meta.forecast_band_c as number[][] | null) || null;
  const tempChartData = useMemo(() => {
    const base = history.map((t, i) => ({
      time: timeLabel(t.timestamp),
      current_temp: t.current_temp as number | undefined,
      lstm_prediction: t.lstm_prediction as number | undefined,
      // anchor the projection line to the latest actual point so it connects
      projection: i === history.length - 1 ? t.current_temp : (undefined as number | undefined),
      bandLo: undefined as number | undefined,
      bandHi: undefined as number | undefined,
    }));
    if (fcTraj && fcTraj.length && history.length) {
      const lastTs = history[history.length - 1].timestamp;
      fcTraj.forEach((v, i) => {
        base.push({
          time: timeLabel(lastTs + i + 1),
          current_temp: undefined, lstm_prediction: undefined, projection: v,
          bandLo: fcBand?.[i]?.[0], bandHi: fcBand?.[i]?.[1],
        });
      });
    }
    return base;
  }, [history, fcTraj, fcBand]);

  // KPIs
  const kpis = useMemo(() => {
    if (!history.length) return { peak: "--", events: 0, actions: 0, latency: "--" };
    let peak = -Infinity, events = 0, actions = 0, latSum = 0;
    let prevAnom: string | null = null;
    for (const t of history) {
      peak = Math.max(peak, t.current_temp);
      latSum += t.latency_ms;
      if (t.action && t.action !== "do-nothing") actions++;
      if (t.active_anomaly && t.active_anomaly !== prevAnom) events++;
      prevAnom = t.active_anomaly;
    }
    return {
      peak: peak === -Infinity ? "--" : `${peak.toFixed(1)}°`,
      events,
      actions,
      latency: `${(latSum / history.length).toFixed(0)}ms`,
    };
  }, [history]);

  // Event log
  const alarms = useMemo(() => {
    const out: { id: number; time: string; sev: string; msg: string }[] = [];
    history.forEach((t) => {
      const sev = t.urgency >= 0.8 ? "critical" : t.urgency >= 0.5 ? "warning" : null;
      const ft = t.metadata?.fault_type;
      const isFault = ft && ft !== "Normal Operations";
      if (sev || isFault) {
        out.push({
          id: t.timestamp,
          time: timeLabel(t.timestamp),
          sev: sev || "warning",
          msg: isFault ? `Fault: ${ft}` : `High urgency ${t.urgency.toFixed(2)} → ${decisionInfo(t.action).text}`,
        });
      }
    });
    const dedup = out.filter((a, idx) => idx === 0 || a.msg !== out[idx - 1].msg);
    return dedup.slice(-7).reverse();
  }, [history]);

  const maintColor =
    maint?.urgency_level === "critical" ? "#f87171"
    : maint?.urgency_level === "high" ? "#fb923c"
    : maint?.urgency_level === "medium" ? "#fbbf24"
    : maint?.urgency_level === "low" ? "#eab308"
    : "#34d399";

  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      {/* Header */}
      <PageHeader icon={Activity} title="Operations Overview" subtitle={latest?.machine_type ? `Monitoring · ${latest.machine_type}` : "Real-time machine health monitoring"}>
        <Badge tone={liveMode ? "bad" : "info"} className={liveMode ? "" : ""}>
          <span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: liveMode ? "#f87171" : "#22d3ee", color: liveMode ? "#f87171" : "#22d3ee" }} />
          {liveMode ? "LIVE · Real HW" : "DEMO · Sim"}
        </Badge>
        <Badge tone={isConnected ? "good" : "bad"}>{isConnected ? "Link OK" : "No Link"}</Badge>
        {systemStatus?.model_status && (
          <Badge
            tone={degraded.length === 0 ? "good" : "warn"}
            title={degraded.length === 0 ? "All models running their trained path." : `Fallback: ${degraded.join(", ")}`}
          >
            {degraded.length === 0 ? "Models ✓" : `${degraded.length} fallback`}
          </Badge>
        )}
        <span className="metric text-xs text-slate-500 px-2">{clock}</span>
        {can("download_report") && (
          <button
            onClick={() => downloadReport()}
            title="Open a printable diagnostic report (save as PDF)."
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border border-teal-400/30 text-teal-300 bg-teal-400/10 hover:bg-teal-400/20 transition-colors"
          >
            <FileDown className="w-3.5 h-3.5" /> Report
          </button>
        )}
      </PageHeader>

      {/* Critical alerts */}
      {(fault || drift) && (
        <div className="flex flex-wrap gap-3">
          {fault && (
            <Card className="!p-3 flex items-center gap-2.5 border-rose-500/30 bg-rose-500/5" title="The fusion layer matched this failure signature.">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              <span className="text-sm text-rose-200">
                Identified fault: <b className="font-semibold">{fault}</b>
                {faultConf ? <span className="text-rose-300/70"> · {Math.round(faultConf * 100)}% confidence</span> : null}
              </span>
            </Card>
          )}
          {drift && (
            <Card className="!p-3 flex items-center gap-2.5 border-amber-500/30 bg-amber-500/5" title="The normal operating range is slowly shifting over time.">
              <Radio className="w-4 h-4 text-amber-400" />
              <span className="text-sm text-amber-200">Concept drift — operating range shifting</span>
            </Card>
          )}
        </div>
      )}

      {/* Hero stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Thermometer}
          label={liveMode ? "Temperature (est.)" : "Temperature"}
          value={temp != null ? temp.toFixed(1) : "--"}
          unit="°C"
          color={tColor}
          sub={liveMode ? `estimated from live CPU load · safe ${th.safe_max}°` : `safe ${th.safe_max}° · crit ${th.critical}°`}
          spark={tempSpark}
          title={liveMode ? "Estimated from real CPU load (Apple Silicon exposes no temp sensor)." : "Live temperature. Green = safe, amber = hot, red = over limit."}
        />
        <StatCard
          icon={HeartPulse}
          label="Machine Health"
          value={health != null ? Math.round(health) : "--"}
          unit="/ 100"
          color={hColor}
          sub={`${healthLabel(health)} · trend ${meta.health_trend || "—"}`}
          title="Composite digital-twin health: thermal, mechanical and longevity."
        />
        <StatCard
          icon={Cpu}
          label="AI Decision"
          value={decision.text}
          color={decision.color}
          sub={`confidence ${(latest?.urgency ?? 0).toFixed(2)}`}
          title={decision.tip}
        />
        <StatCard
          icon={Timer}
          label="Time to Limit"
          value={rulText}
          color={rulColor}
          sub={latest ? `${latest.latency_ms.toFixed(1)} ms response` : "—"}
          title="Estimated time before the machine reaches its safe limit."
        />
      </div>

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-3">
        <KpiTile icon={Clock} label="Session Uptime" value={fmtUptime(uptime)} tip="How long this monitoring session has been open." />
        <KpiTile icon={TrendingUp} label="Peak Temp (2m)" value={kpis.peak} tip="Highest temperature in the last ~2 minutes." />
        <KpiTile icon={Bell} label="Events (2m)" value={String(kpis.events)} tip="Anomaly episodes in the last ~2 minutes." />
        <KpiTile icon={Zap} label="Actions (2m)" value={String(kpis.actions)} tip="Protective actions taken recently." />
        <KpiTile icon={Timer} label="Avg Response" value={kpis.latency} tip="Average model response time (lower is better)." />
      </div>

      {/* Temperature chart + Where AI runs */}
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-8">
          <CardTitle
            icon={Thermometer}
            right={
              <div className="flex items-center gap-3 text-[11px] text-slate-500">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 rounded" style={{ background: CHART.temp }} /> Now</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 rounded" style={{ background: CHART.forecast }} /> Forecast</span>
                {fcTraj && <span className="flex items-center gap-1.5"><span className="w-2.5 h-0.5 rounded" style={{ background: CHART.indigo }} /> Projection +{fcTraj.length}s</span>}
              </div>
            }
          >
            Temperature — Now, Forecast & Projection
          </CardTitle>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={tempChartData} margin={{ top: 6, right: 12, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="tempFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={CHART.temp} stopOpacity={0.18} />
                    <stop offset="100%" stopColor={CHART.temp} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="time" stroke={CHART.axis} fontSize={CHART.tickFont} tickMargin={8} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART.axis} fontSize={CHART.tickFont} domain={["auto", "auto"]} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                <ReferenceLine y={th.safe_max} stroke={CHART.warn} strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "SAFE", fill: CHART.warn, fontSize: 9, position: "insideTopRight" }} />
                <ReferenceLine y={th.critical} stroke={CHART.crit} strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "LIMIT", fill: CHART.crit, fontSize: 9, position: "insideTopRight" }} />
                {/* P2.1: p10 / p90 uncertainty band as two faint dotted bounds (keeps axis tight) */}
                <Line type="monotone" dataKey="bandHi" name="p90" stroke={CHART.indigo} strokeWidth={1} strokeOpacity={0.4} strokeDasharray="1 3" dot={false} isAnimationActive={false} connectNulls legendType="none" />
                <Line type="monotone" dataKey="bandLo" name="p10" stroke={CHART.indigo} strokeWidth={1} strokeOpacity={0.4} strokeDasharray="1 3" dot={false} isAnimationActive={false} connectNulls legendType="none" />
                <Area type="monotone" dataKey="current_temp" name="Now (°C)" stroke={CHART.temp} strokeWidth={2.2} fill="url(#tempFill)" dot={false} isAnimationActive={false} connectNulls={false} />
                <Line type="monotone" dataKey="lstm_prediction" name="Forecast (°C)" stroke={CHART.forecast} strokeWidth={1.6} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="projection" name="Projection (°C)" stroke={CHART.indigo} strokeWidth={1.8} strokeDasharray="2 3" dot={false} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Live vitals (system_health components — real per-component values) */}
        <Card className="col-span-12 lg:col-span-4">
          <CardTitle icon={Gauge} right={<span className="metric text-xs" style={{ color: hColor }}>{health != null ? Math.round(health) : "--"}/100</span>}>
            Live Vitals
          </CardTitle>
          <div className="space-y-2.5">
            {(sh?.components || []).slice(0, 5).map((c) => (
              <div key={c.name} className="flex items-center gap-2.5" title={c.verdict}>
                <span className="text-base w-5 text-center">{c.icon}</span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-300">{c.name}</span>
                    <span className="metric text-xs text-slate-100">{c.val}</span>
                  </div>
                  <div className="text-[10px]" style={{ color: statusColor(c.status) }}>{c.verdict}</div>
                </div>
              </div>
            ))}
            {!sh && <Awaiting />}
          </div>
        </Card>
      </div>

      {/* Anomaly signals + Why this decision */}
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-7">
          <CardTitle
            icon={Activity}
            right={
              <div className="flex flex-wrap gap-2.5 text-[10px] text-slate-500">
                <span style={{ color: CHART.cyan }}>● Odd</span>
                <span style={{ color: CHART.good }}>● Trend</span>
                <span style={{ color: CHART.amber }}>● Pattern</span>
                <span style={{ color: CHART.fuchsia }}>● Vibration</span>
                {!detectorGated && <span style={{ color: CHART.bad }}>● Detector{foundationOn ? " (0-shot)" : ""}</span>}
              </div>
            }
          >
            Anomaly Signals (per model, 0–1)
            {detectorP != null && (
              <span className="ml-2 text-[11px]" style={{ color: detectorP > 0.5 ? CHART.bad : CHART.good }}>
                · P(fault) {(detectorP * 100).toFixed(0)}%
              </span>
            )}
            {detectorGated && <span className="ml-2 text-[11px] text-slate-500">· detector calibrated for simulation</span>}
            {foundationOn && <span className="ml-2 text-[11px] text-indigo-300/80">· zero-shot foundation detector</span>}
          </CardTitle>
          <div className="h-[230px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 6, right: 12, left: -24, bottom: 0 }}>
                <defs>
                  <linearGradient id="combFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={CHART.combined} stopOpacity={0.12} />
                    <stop offset="100%" stopColor={CHART.combined} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="time" stroke={CHART.axis} fontSize={CHART.tickFont} tickMargin={8} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART.axis} fontSize={CHART.tickFont} domain={[0, 1]} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                {!detectorGated && <Area type="monotone" dataKey="detector" name="Detector P(fault)" stroke={CHART.bad} strokeWidth={2.2} fill="url(#combFill)" dot={false} isAnimationActive={false} />}
                <Line type="monotone" dataKey="if_score" name="Odd reading" stroke={CHART.cyan} strokeWidth={1.3} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="lstm_score" name="Heat trend" stroke={CHART.good} strokeWidth={1.3} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="ae_score" name="Pattern" stroke={CHART.amber} strokeWidth={1.3} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="vib_score" name="Vibration" stroke={CHART.fuchsia} strokeWidth={1.3} dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-5">
          <CardTitle icon={Sparkles}>Why This Decision?</CardTitle>
          <div className="space-y-3">
            {XAI_ROWS.map((r) => {
              const v = contrib[r.key] ?? 0;
              const active = v > 25;
              return (
                <div key={r.key} className="flex items-center gap-3" title={`${r.meaning} — ${r.model}`}>
                  <div className="w-28 shrink-0">
                    <div className="text-xs text-slate-300">{r.label}</div>
                    <div className="text-[10px] text-slate-600">{r.meaning}</div>
                  </div>
                  <div className="flex-1"><Progress value={v} color={active ? "#22d3ee" : "#475569"} /></div>
                  <span className="metric text-xs w-9 text-right" style={{ color: active ? "#22d3ee" : "#5d6b82" }}>{v}%</span>
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-slate-500 mt-4 pt-3 border-t border-white/[0.06]">
            {topRow ? (
              <><span className="text-cyan-400 font-medium">Main reason:</span> {topRow.label} — {topRow.meaning.toLowerCase()}</>
            ) : (
              <span className="text-slate-600">No single cause — system looks normal.</span>
            )}
          </p>
        </Card>
      </div>

      {/* Maintenance + Inject panel */}
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-5">
          <CardTitle icon={Wrench}>Maintenance Plan</CardTitle>
          {maint ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="metric text-lg" style={{ color: maintColor }}>{maint.status?.toUpperCase()}</span>
                <span className="pill" style={{ color: maintColor, borderColor: `${maintColor}55`, background: `${maintColor}14` }}>
                  {maint.urgency_level?.toUpperCase()}
                </span>
              </div>
              <p className="text-[13px] text-slate-400 leading-relaxed">{maint.action}</p>
              {maint.window_start && (
                <p className="text-[11px] text-slate-500">
                  Window: {new Date(maint.window_start).toLocaleString()} → {maint.window_end ? new Date(maint.window_end).toLocaleString() : "—"}
                </p>
              )}
            </div>
          ) : (
            <Awaiting />
          )}
        </Card>

        <Card className="col-span-12 lg:col-span-7">
          <CardTitle icon={ShieldAlert}>Test Panel · Inject a Fault</CardTitle>
          <p className="text-[11px] text-slate-500 mb-3">
            {can("inject") ? "Click to simulate a fault and watch the AI react." : "Read-only — sign in as operator to use."}
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {ANOMALY_BUTTONS.map((b) => (
              <button
                key={b.type}
                onClick={() => injectAnomaly(b.type)}
                disabled={!can("inject")}
                title={b.tip}
                className="flex items-center gap-2 rounded-xl border px-3 py-2.5 text-xs font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed hover:bg-white/[0.04] hover:-translate-y-0.5"
                style={{ borderColor: `${b.color}40`, color: b.color }}
              >
                <b.icon className="w-3.5 h-3.5 shrink-0" /> {b.label}
              </button>
            ))}
            <button
              onClick={resetSystem}
              disabled={!can("inject")}
              title="Clear all injected faults and return to idle."
              className="flex items-center gap-2 rounded-xl border border-white/[0.1] text-slate-400 px-3 py-2.5 text-xs font-medium hover:bg-white/[0.04] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw className="w-3.5 h-3.5 shrink-0" /> Reset
            </button>
          </div>
        </Card>
      </div>

      {/* Event log */}
      <Card>
        <CardTitle icon={Bell}>Event Log</CardTitle>
        <div className="space-y-1.5 max-h-[200px] overflow-y-auto pr-1">
          {alarms.length === 0 && <p className="text-sm text-slate-600 py-2">No active alarms — all clear.</p>}
          {alarms.map((a) => {
            const ackd = acked.has(a.id);
            const col = a.sev === "critical" ? "#f87171" : "#fbbf24";
            return (
              <div key={a.id} className="flex items-center gap-3 rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-2">
                <span className="metric text-[11px] text-slate-500 w-16 shrink-0">{a.time}</span>
                <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0" style={{ background: ackd ? "#475569" : col }} />
                <span className={`text-xs flex-1 truncate ${ackd ? "text-slate-600 line-through" : "text-slate-300"}`}>{a.msg}</span>
                {!ackd && (
                  <button
                    onClick={() => setAcked((s) => new Set(s).add(a.id))}
                    title="Acknowledge"
                    className="flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] border border-white/[0.1] text-slate-400 hover:text-cyan-300 hover:border-cyan-500/40"
                  >
                    <Check className="w-3 h-3" /> Ack
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </Card>

      {/* AI assistant alert */}
      <Card accent>
        <CardTitle icon={Activity} right={latest?.llm_source ? <Badge tone="indigo">{latest.llm_source}</Badge> : undefined}>
          AI Assistant — Plain-Language Alert
        </CardTitle>
        <div className="rounded-xl bg-black/30 border border-white/[0.06] p-4 font-mono text-[13px] leading-relaxed min-h-[60px]">
          {latest ? (
            <span className={latest.urgency > 0.6 ? "text-rose-300" : "text-emerald-300"}>
              <span className="text-slate-600">$ </span>{latest.alert}
            </span>
          ) : (
            <span className="text-slate-600">awaiting telemetry sequence…</span>
          )}
        </div>
      </Card>
    </div>
  );
}
