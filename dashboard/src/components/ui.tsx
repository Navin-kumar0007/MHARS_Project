"use client";

import React from "react";
import { ResponsiveContainer, AreaChart, Area } from "recharts";

// ── classnames helper ──────────────────────────────────────────────────────
export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

// ── Shared chart theme ───────────────────────────────────────────────────────
export const CHART = {
  grid: "#1b2433",
  axis: "#3c4a60",
  tickFont: 10,
  // Semantic + model palette (kept consistent across every page)
  temp: "#f87171",
  forecast: "#22d3ee",
  good: "#34d399",
  warn: "#fbbf24",
  bad: "#f87171",
  crit: "#ef4444",
  teal: "#2dd4bf",
  cyan: "#22d3ee",
  indigo: "#818cf8",
  amber: "#fbbf24",
  fuchsia: "#e879f9",
  combined: "#e8edf6",
};

export const tooltipStyle = {
  backgroundColor: "rgba(10,14,26,0.95)",
  border: "1px solid rgba(148,163,184,0.18)",
  borderRadius: 10,
  fontSize: 12,
  boxShadow: "0 12px 32px -12px rgba(0,0,0,0.8)",
  padding: "8px 12px",
} as const;

export const tooltipLabelStyle = { color: "#94a3b8", fontSize: 11, marginBottom: 2 } as const;

// ── Health helpers (single source of truth across the app) ───────────────────
export function healthColor(score: number | null | undefined): string {
  if (score == null) return "#5d6b82";
  if (score >= 75) return "#34d399";
  if (score >= 45) return "#fbbf24";
  return "#f87171";
}
export function healthLabel(score: number | null | undefined): string {
  if (score == null) return "—";
  if (score >= 75) return "Healthy";
  if (score >= 45) return "Degraded";
  return "Critical";
}

// ── Page header ──────────────────────────────────────────────────────────────
export function PageHeader({
  icon: Icon,
  title,
  subtitle,
  accent = "#2dd4bf",
  children,
}: {
  icon: React.ElementType;
  title: string;
  subtitle?: string;
  accent?: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-4 flex-wrap pb-1">
      <div className="flex items-center gap-3">
        <div
          className="grid place-items-center w-10 h-10 rounded-xl shrink-0"
          style={{ background: `${accent}1a`, border: `1px solid ${accent}33` }}
        >
          <Icon className="w-5 h-5" style={{ color: accent }} />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-slate-100 tracking-tight">{title}</h1>
          {subtitle && <p className="text-[13px] text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {children && <div className="flex items-center gap-2 flex-wrap">{children}</div>}
    </header>
  );
}

// ── Card ───────────────────────────────────────────────────────────────────
export function Card({
  className = "",
  hover = false,
  accent = false,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement> & { hover?: boolean; accent?: boolean }) {
  return (
    <div className={cx("card p-5", hover && "card-hover", accent && "card-accent", className)} {...rest}>
      {children}
    </div>
  );
}

// ── Card title row ───────────────────────────────────────────────────────────
export function CardTitle({
  icon: Icon,
  children,
  right,
  className = "",
}: {
  icon?: React.ElementType;
  children: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cx("flex items-center justify-between gap-2 mb-4", className)}>
      <div className="flex items-center gap-2 eyebrow">
        {Icon && <Icon className="w-3.5 h-3.5 text-slate-400" />}
        {children}
      </div>
      {right}
    </div>
  );
}

// ── Badge ──────────────────────────────────────────────────────────────────
const TONES: Record<string, string> = {
  good: "text-emerald-300 border-emerald-500/30 bg-emerald-500/10",
  warn: "text-amber-300 border-amber-500/30 bg-amber-500/10",
  bad: "text-rose-300 border-rose-500/30 bg-rose-500/10",
  info: "text-cyan-300 border-cyan-500/30 bg-cyan-500/10",
  indigo: "text-indigo-300 border-indigo-500/30 bg-indigo-500/10",
  neutral: "text-slate-300 border-slate-600/40 bg-slate-500/10",
};
export function Badge({
  tone = "neutral",
  children,
  className = "",
  title,
}: {
  tone?: keyof typeof TONES | string;
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  return <span title={title} className={cx("pill", TONES[tone] || TONES.neutral, className)}>{children}</span>;
}

// ── Stat card ────────────────────────────────────────────────────────────────
export function StatCard({
  icon: Icon,
  label,
  value,
  unit,
  sub,
  color = "#e8edf6",
  spark,
  title,
}: {
  icon: React.ElementType;
  label: string;
  value: React.ReactNode;
  unit?: string;
  sub?: React.ReactNode;
  color?: string;
  spark?: number[];
  title?: string;
}) {
  return (
    <Card hover className="flex flex-col gap-2 overflow-hidden" title={title}>
      <div className="flex items-center justify-between">
        <span className="eyebrow">{label}</span>
        <Icon className="w-4 h-4" style={{ color }} />
      </div>
      <div className="flex items-end gap-1.5">
        <span className="metric text-[28px] leading-none" style={{ color }}>
          {value}
        </span>
        {unit && <span className="text-sm text-slate-500 mb-0.5">{unit}</span>}
      </div>
      {sub && <div className="text-[11px] text-slate-500 truncate">{sub}</div>}
      {spark && spark.length > 1 && (
        <div className="h-7 -mx-1 -mb-1 mt-0.5">
          <Sparkline data={spark} color={color} />
        </div>
      )}
    </Card>
  );
}

// ── Sparkline ────────────────────────────────────────────────────────────────
export function Sparkline({ data, color = "#2dd4bf" }: { data: number[]; color?: string }) {
  const d = data.map((v, i) => ({ i, v }));
  const id = React.useId().replace(/:/g, "");
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={d} margin={{ top: 2, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id={`spark-${id}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.35} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area
          type="monotone"
          dataKey="v"
          stroke={color}
          strokeWidth={1.5}
          fill={`url(#spark-${id})`}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Progress bar ─────────────────────────────────────────────────────────────
export function Progress({ value, color = "#2dd4bf", track = "rgba(148,163,184,0.12)" }: { value: number; color?: string; track?: string }) {
  return (
    <div className="h-1.5 w-full rounded-full overflow-hidden" style={{ background: track }}>
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }}
      />
    </div>
  );
}

// ── Empty / loading state ────────────────────────────────────────────────────
export function Awaiting({ label = "Waiting for telemetry…" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-slate-600">
      <span className="w-2 h-2 rounded-full bg-cyan-400 pulse-dot text-cyan-400" />
      <span className="text-sm">{label}</span>
    </div>
  );
}
