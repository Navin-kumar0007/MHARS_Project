"use client";

import React from "react";
import { useTelemetry } from "@/components/TelemetryProvider";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
  Cell,
} from "recharts";
import { BarChart3, HeartPulse, Timer, Layers, TrendingUp, Sigma } from "lucide-react";
import { Card, CardTitle, PageHeader, Awaiting, CHART, tooltipStyle, tooltipLabelStyle, healthColor } from "@/components/ui";

function timeLabel(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour12: false, minute: "2-digit", second: "2-digit" });
}

function HealthRing({ score }: { score: number }) {
  const r = 54;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const color = healthColor(score);
  return (
    <div className="relative flex items-center justify-center w-36 h-36">
      <svg viewBox="0 0 140 140" className="w-36 h-36 -rotate-90">
        <circle cx="70" cy="70" r={r} fill="none" stroke="rgba(148,163,184,0.12)" strokeWidth="12" />
        <circle
          cx="70" cy="70" r={r} fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - pct)}
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.6s ease", filter: `drop-shadow(0 0 6px ${color}80)` }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="metric text-3xl" style={{ color }}>{Math.round(score)}</span>
        <span className="text-[10px] text-slate-500 uppercase tracking-wider">/ 100</span>
      </div>
    </div>
  );
}

function rulText(mins: number | null | undefined) {
  if (mins == null) return { d: "—", h: "—", m: "—" };
  const total = Math.max(0, mins);
  return {
    d: String(Math.floor(total / 1440)),
    h: String(Math.floor((total % 1440) / 60)).padStart(2, "0"),
    m: String(Math.floor(total % 60)).padStart(2, "0"),
  };
}

export default function AnalyticsPage() {
  const { latest, history } = useTelemetry();
  const meta = latest?.metadata || {};

  const health = (meta.health_score ?? 0) as number;
  const breakdown = (meta.health_breakdown || {}) as Record<string, number>;
  const trendStats = meta.trend_stats || {};
  const featImp = (meta.feature_importance || {}) as Record<string, number>;
  const rul = meta.rul_minutes as number | null | undefined;
  const interval = meta.prediction_interval || {};
  const rt = rulText(rul);

  const trendData = history.map((t) => ({
    time: timeLabel(t.timestamp),
    cusum: t.metadata?.trend_stats?.cusum ?? 0,
    ewma: t.metadata?.trend_stats?.ewma ?? 0,
  }));

  const uncData = history.map((t) => {
    const pi = t.metadata?.prediction_interval || {};
    const lo = pi.lower, up = pi.upper;
    return {
      time: timeLabel(t.timestamp),
      temp: t.current_temp,
      pred: t.lstm_prediction,
      lower: lo,
      band: lo !== undefined && up !== undefined ? up - lo : undefined,
    };
  });

  const featData = Object.entries(featImp).map(([name, val]) => ({ name, val }));
  const breakdownData = Object.entries(breakdown).map(([name, val]) => ({ name, val }));
  const featColors = [CHART.cyan, CHART.teal, CHART.amber, CHART.indigo, CHART.fuchsia, "#60a5fa"];

  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      <PageHeader icon={BarChart3} title="Advanced Analytics" subtitle="Health, trend drift, explainability and uncertainty" accent="#818cf8" />

      {!latest && <Awaiting />}

      {latest && (
        <>
          <div className="grid grid-cols-12 gap-4">
            <Card className="col-span-12 md:col-span-4 flex flex-col items-center">
              <CardTitle icon={HeartPulse} className="self-start w-full">Composite Health</CardTitle>
              <HealthRing score={health} />
              <p className="text-[11px] text-slate-500 mt-3">Trend: <span className="text-slate-300">{meta.health_trend || "—"}</span></p>
            </Card>

            <Card className="col-span-12 md:col-span-4">
              <CardTitle icon={Layers}>Health Breakdown</CardTitle>
              <div className="h-[150px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={breakdownData} layout="vertical" margin={{ left: 10, right: 20 }}>
                    <XAxis type="number" domain={[0, 100]} stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="name" stroke={CHART.axis} fontSize={10} width={72} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                    <Bar dataKey="val" radius={[0, 5, 5, 0]} isAnimationActive={false}>
                      {breakdownData.map((d, i) => (
                        <Cell key={i} fill={healthColor(d.val)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card className="col-span-12 md:col-span-4 flex flex-col">
              <CardTitle icon={Timer}>Remaining Useful Life</CardTitle>
              <div className="flex-1 flex items-center justify-center gap-4">
                {[{ v: rt.d, l: "days" }, { v: rt.h, l: "hrs" }, { v: rt.m, l: "min" }].map((b) => (
                  <div key={b.l} className="text-center">
                    <div className="metric text-4xl text-teal-300">{b.v}</div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{b.l}</div>
                  </div>
                ))}
              </div>
              <p className="text-[11px] text-slate-500 mt-3 text-center">
                {rul == null ? "System stable — no degradation trend" : `≈ ${Math.round(rul)} minutes to threshold`}
              </p>
            </Card>
          </div>

          <div className="grid grid-cols-12 gap-4">
            <Card className="col-span-12 lg:col-span-5">
              <CardTitle icon={Sigma}>Explainability — Feature Attribution</CardTitle>
              <div className="h-[260px]">
                {featData.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={featData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                      <XAxis dataKey="name" stroke={CHART.axis} fontSize={9} interval={0} angle={-20} textAnchor="end" height={50} tickLine={false} axisLine={false} />
                      <YAxis stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                      <Bar dataKey="val" name="Importance %" radius={[5, 5, 0, 0]} isAnimationActive={false}>
                        {featData.map((_, i) => (<Cell key={i} fill={featColors[i % featColors.length]} />))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full grid place-items-center text-slate-600 text-sm">No attribution data</div>
                )}
              </div>
              <p className="text-[11px] text-slate-500 mt-2">
                Top driver: <span className="text-slate-300">{meta.top_contributor || "—"}</span> · Fault: <span className="text-slate-300">{meta.fault_type || "—"}</span>
              </p>
            </Card>

            <Card className="col-span-12 lg:col-span-7">
              <CardTitle icon={TrendingUp}>LSTM Prediction with Uncertainty Band</CardTitle>
              <div className="h-[260px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={uncData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                    <XAxis dataKey="time" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis stroke={CHART.axis} fontSize={10} domain={["auto", "auto"]} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                    <Area type="monotone" dataKey="lower" stackId="ci" stroke="none" fill="transparent" isAnimationActive={false} legendType="none" />
                    <Area type="monotone" dataKey="band" stackId="ci" name="95% CI" stroke="none" fill={CHART.teal} fillOpacity={0.15} isAnimationActive={false} />
                    <Line type="monotone" dataKey="temp" name="Actual °C" stroke={CHART.temp} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="pred" name="LSTM Pred" stroke={CHART.teal} strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <p className="text-[11px] text-slate-500 mt-2">
                Interval: {interval.lower ?? "—"}°C – {interval.upper ?? "—"}°C · Confidence {interval.confidence_score ?? meta.urgency_confidence ?? "—"}
              </p>
            </Card>
          </div>

          <Card>
            <CardTitle icon={TrendingUp}>CUSUM / EWMA Drift Control Chart</CardTitle>
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="time" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={5} stroke={CHART.crit} strokeDasharray="4 4" strokeOpacity={0.6} label={{ value: "CUSUM h=5", fill: CHART.crit, fontSize: 10, position: "right" }} />
                  <Line type="monotone" dataKey="cusum" name="CUSUM" stroke={CHART.amber} strokeWidth={2} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="ewma" name="EWMA" stroke={CHART.cyan} strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-500 mt-2">
              Current drift: <span className={trendStats.is_drifting ? "text-rose-400" : "text-emerald-400"}>{trendStats.is_drifting ? "DRIFTING" : "stable"}</span>
              {" "}· trend score {typeof trendStats.trend_score === "number" ? trendStats.trend_score.toFixed(2) : "—"}
            </p>
          </Card>
        </>
      )}
    </div>
  );
}
