"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useTelemetry, API_BASE } from "@/components/TelemetryProvider";
import { Card, CardTitle, PageHeader, Badge, Progress, Awaiting, CHART } from "@/components/ui";
import { Activity, Boxes, Gauge, Timer, Cpu, ShieldCheck } from "lucide-react";

type EvalReport = {
  generated_at: number;
  samples: number;
  positives: number;
  detectors: Record<string, { p: number; r: number; f1: number; best_f1: number; best_thr: number; roc_auc: number }>;
  per_fault: Record<string, { n: number; detected: number }>;
};

export default function DiagnosticsPage() {
  const { systemStatus, latest, history, isConnected } = useTelemetry();
  const [evalRep, setEvalRep] = useState<EvalReport | null>(null);
  const [evalAvail, setEvalAvail] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/eval_report`)
      .then((r) => r.json())
      .then((d) => { setEvalAvail(!!d.available); setEvalRep(d.report); })
      .catch(() => setEvalAvail(false));
  }, []);

  const ms = systemStatus?.model_status || {};
  const degraded = systemStatus?.models_degraded || [];

  const latency = useMemo(() => {
    if (!history.length) return { avg: "--", min: "--", max: "--" };
    const l = history.map((t) => t.latency_ms);
    return {
      avg: `${(l.reduce((a, b) => a + b, 0) / l.length).toFixed(1)}ms`,
      min: `${Math.min(...l).toFixed(0)}ms`,
      max: `${Math.max(...l).toFixed(0)}ms`,
    };
  }, [history]);

  const detP = latest?.metadata?.anomaly_probability as number | undefined;

  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      <PageHeader icon={Activity} title="Diagnostics" subtitle="Model provenance, live performance and detection metrics" accent="#818cf8">
        <Badge tone={degraded.length === 0 ? "good" : "warn"}>
          {degraded.length === 0 ? "All models live" : `${degraded.length} in fallback`}
        </Badge>
      </PageHeader>

      {/* Live performance */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card hover><div className="eyebrow">Inference Latency (avg)</div><div className="metric text-2xl mt-1 text-teal-300">{latency.avg}</div><div className="text-[11px] text-slate-500 mt-0.5">min {latency.min} · max {latency.max}</div></Card>
        <Card hover><div className="eyebrow">Telemetry</div><div className="metric text-2xl mt-1" style={{ color: isConnected ? CHART.good : CHART.bad }}>{isConnected ? "Live" : "Down"}</div><div className="text-[11px] text-slate-500 mt-0.5">{history.length} samples buffered</div></Card>
        <Card hover><div className="eyebrow">Detector P(fault)</div><div className="metric text-2xl mt-1" style={{ color: detP != null && detP > 0.5 ? CHART.bad : CHART.good }}>{detP != null ? `${(detP * 100).toFixed(0)}%` : "--"}</div><div className="text-[11px] text-slate-500 mt-0.5">supervised classifier</div></Card>
        <Card hover><div className="eyebrow">Active Machine</div><div className="metric text-2xl mt-1 text-slate-100">{latest?.machine_type || "—"}</div><div className="text-[11px] text-slate-500 mt-0.5">{latest?.live_mode ? "live hardware" : "simulation"}</div></Card>
      </div>

      {/* Model provenance */}
      <Card>
        <CardTitle icon={Boxes}>Model Provenance</CardTitle>
        {Object.keys(ms).length === 0 ? <Awaiting label="Loading model status…" /> : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(ms).map(([name, v]) => (
              <div key={name} className="flex items-center justify-between rounded-xl border border-white/[0.06] bg-white/[0.02] px-3.5 py-2.5">
                <div className="flex items-center gap-2.5">
                  <ShieldCheck className="w-4 h-4" style={{ color: v.ok ? CHART.good : CHART.warn }} />
                  <span className="text-sm text-slate-200">{name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-500">{v.detail}</span>
                  <Badge tone={v.ok ? "good" : "warn"}>{v.ok ? "live" : "fallback"}</Badge>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Eval metrics */}
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-7">
          <CardTitle icon={Gauge} right={evalRep ? <span className="text-[11px] text-slate-500">{evalRep.samples} samples · {evalRep.positives} faults</span> : undefined}>
            Anomaly Detection — Offline Eval
          </CardTitle>
          {evalAvail === false && (
            <p className="text-sm text-slate-500">No eval report yet. Run <span className="metric text-slate-300">python3 tools/eval_anomaly.py</span> to generate one.</p>
          )}
          {evalRep && (
            <div className="space-y-2">
              <div className="grid grid-cols-[1.6fr_repeat(3,1fr)] gap-2 eyebrow px-1">
                <span>Detector</span><span className="text-right">F1</span><span className="text-right">best F1</span><span className="text-right">ROC-AUC</span>
              </div>
              {Object.entries(evalRep.detectors).map(([name, d]) => {
                const strong = d.roc_auc >= 0.8;
                return (
                  <div key={name} className="grid grid-cols-[1.6fr_repeat(3,1fr)] gap-2 items-center rounded-lg border border-white/[0.05] bg-white/[0.02] px-3 py-1.5">
                    <span className="text-xs text-slate-300">{name}</span>
                    <span className="metric text-xs text-right text-slate-400">{d.f1.toFixed(2)}</span>
                    <span className="metric text-xs text-right text-slate-400">{d.best_f1.toFixed(2)}</span>
                    <span className="metric text-xs text-right font-semibold" style={{ color: strong ? CHART.good : CHART.warn }}>{d.roc_auc.toFixed(2)}</span>
                  </div>
                );
              })}
              <p className="text-[11px] text-slate-500 pt-1">Higher ROC-AUC = better separation. The supervised classifier P(fault) is the adopted detector.</p>
            </div>
          )}
          {evalAvail === null && <Awaiting label="Loading eval report…" />}
        </Card>

        <Card className="col-span-12 lg:col-span-5">
          <CardTitle icon={ShieldCheck}>Per-Fault Detection</CardTitle>
          {evalRep ? (
            <div className="space-y-2.5">
              {Object.entries(evalRep.per_fault).map(([f, d]) => (
                <div key={f} className="flex items-center gap-3">
                  <span className="text-xs text-slate-300 w-32 capitalize">{f.replace(/_/g, " ")}</span>
                  <div className="flex-1"><Progress value={d.detected * 100} color={d.detected >= 0.7 ? CHART.good : CHART.warn} /></div>
                  <span className="metric text-xs w-10 text-right text-slate-300">{(d.detected * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          ) : (
            evalAvail === false ? <p className="text-sm text-slate-500">—</p> : <Awaiting label="…" />
          )}
        </Card>
      </div>
    </div>
  );
}
