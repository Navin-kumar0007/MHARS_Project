"use client";

import React from "react";
import { ResponsiveContainer, AreaChart, Area, RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";

// ── classnames helper ──────────────────────────────────────────────────────
export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

// ── Shared chart theme (tuned for LIGHT background) ──────────────────────────
export const CHART = {
  grid: "rgba(16,24,40,0.07)",
  axis: "#9aa0ad",
  tickFont: 10,
  temp: "#ef4444",
  forecast: "#0ea5e9",
  good: "#10b981",
  warn: "#f59e0b",
  bad: "#ef4444",
  crit: "#dc2626",
  accent: "#5e6ad2",
  teal: "#5e6ad2",   // alias kept for back-compat; now indigo
  cyan: "#0ea5e9",
  indigo: "#5e6ad2",
  amber: "#f59e0b",
  fuchsia: "#d946ef",
  combined: "#475569",
};

export const tooltipStyle = {
  backgroundColor: "#ffffff",
  border: "1px solid #e7e9ee",
  borderRadius: 12,
  fontSize: 12,
  color: "#0c0e14",
  boxShadow: "0 8px 28px -8px rgba(16,24,40,0.22)",
  padding: "8px 12px",
} as const;

export const tooltipLabelStyle = { color: "#8b91a1", fontSize: 11, marginBottom: 2 } as const;

// ── Health helpers (single source of truth across the app) ───────────────────
export function healthColor(score: number | null | undefined): string {
  if (score == null) return "#8b91a1";
  if (score >= 75) return "#10b981";
  if (score >= 45) return "#f59e0b";
  return "#ef4444";
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
  accent = "#5e6ad2",
  children,
}: {
  icon: React.ElementType;
  title: string;
  subtitle?: string;
  accent?: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="flex items-start justify-between gap-4 flex-wrap pb-1 fade-in">
      <div className="flex items-center gap-3">
        <div
          className="grid place-items-center w-11 h-11 rounded-2xl shrink-0"
          style={{ background: `${accent}14`, border: `1px solid ${accent}33`, boxShadow: `0 6px 16px -8px ${accent}66` }}
        >
          <Icon className="w-5 h-5" style={{ color: accent }} />
        </div>
        <div>
          <h1 className="text-2xl font-bold text-[var(--text)] tracking-tight">{title}</h1>
          {subtitle && <p className="text-[13px] text-[var(--text-dim)] mt-0.5">{subtitle}</p>}
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
  glow = false,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement> & { hover?: boolean; accent?: boolean; glow?: boolean }) {
  return (
    <div className={cx("card p-5", hover && "card-hover", accent && "card-accent", glow && "card-glow", className)} {...rest}>
      {children}
    </div>
  );
}

// ── Skeleton (loading placeholder) ───────────────────────────────────────────
export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={cx("shimmer rounded-lg", className)} />;
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
        {Icon && <Icon className="w-3.5 h-3.5 text-[var(--accent)]" />}
        {children}
      </div>
      {right}
    </div>
  );
}

// ── Badge ──────────────────────────────────────────────────────────────────
const TONES: Record<string, string> = {
  good: "text-emerald-700 border-emerald-200 bg-emerald-50",
  warn: "text-amber-700 border-amber-200 bg-amber-50",
  bad: "text-rose-700 border-rose-200 bg-rose-50",
  info: "text-sky-700 border-sky-200 bg-sky-50",
  indigo: "text-indigo-700 border-indigo-200 bg-indigo-50",
  neutral: "text-slate-600 border-slate-200 bg-slate-50",
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
  color = "#0c0e14",
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
    <Card hover className="group relative flex flex-col gap-3 overflow-hidden" title={title}>
      <span className="absolute left-0 top-5 bottom-5 w-[3px] rounded-full" style={{ background: color }} />
      <div className="flex items-center justify-between pl-1.5">
        <span className="eyebrow">{label}</span>
        <span
          className="grid place-items-center w-9 h-9 rounded-xl shrink-0 transition-transform group-hover:scale-110"
          style={{ background: `${color}16`, border: `1px solid ${color}2e` }}
        >
          <Icon className="w-4 h-4" style={{ color }} />
        </span>
      </div>
      <div className="flex items-end gap-1.5 pl-1.5">
        <span className="metric text-[34px] leading-none tracking-tight" style={{ color }}>
          {value}
        </span>
        {unit && <span className="text-sm text-[var(--text-muted)] mb-1 font-medium">{unit}</span>}
      </div>
      {sub && <div className="text-[11px] text-[var(--text-dim)] truncate pl-1.5">{sub}</div>}
      {spark && spark.length > 1 && (
        <div className="h-9 -mx-1 -mb-1 mt-0.5">
          <Sparkline data={spark} color={color} />
        </div>
      )}
    </Card>
  );
}

// ── Radial gauge (health / score 0–100) ──────────────────────────────────────
export function RadialGauge({
  value,
  color,
  size = 160,
  label,
  sub,
}: {
  value: number;
  color: string;
  size?: number;
  label?: React.ReactNode;
  sub?: React.ReactNode;
}) {
  const v = Math.max(0, Math.min(100, value));
  const data = [{ name: "v", value: v, fill: color }];
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart innerRadius="74%" outerRadius="100%" data={data} startAngle={90} endAngle={-270} barSize={12}>
          <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
          <RadialBar background={{ fill: "#eef0f4" }} dataKey="value" cornerRadius={12} isAnimationActive={false} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 grid place-items-center text-center">
        <div>
          <div className="metric text-3xl leading-none" style={{ color }}>{label ?? Math.round(v)}</div>
          {sub && <div className="text-[11px] text-[var(--text-dim)] mt-1">{sub}</div>}
        </div>
      </div>
    </div>
  );
}

// ── Sparkline ────────────────────────────────────────────────────────────────
export function Sparkline({ data, color = "#5e6ad2" }: { data: number[]; color?: string }) {
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
          strokeWidth={1.75}
          fill={`url(#spark-${id})`}
          isAnimationActive={false}
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ── Progress bar ─────────────────────────────────────────────────────────────
export function Progress({ value, color = "#5e6ad2", track = "#eef0f4" }: { value: number; color?: string; track?: string }) {
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
    <div className="flex flex-col items-center justify-center gap-2 py-16 text-[var(--text-muted)]">
      <span className="w-2 h-2 rounded-full pulse-dot" style={{ background: "#5e6ad2", color: "#5e6ad2" }} />
      <span className="text-sm">{label}</span>
    </div>
  );
}
