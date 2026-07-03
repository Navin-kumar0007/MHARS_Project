"use client";

import React from "react";
import { ResponsiveContainer, AreaChart, Area, RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";

// ── classnames helper ──────────────────────────────────────────────────────
export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(" ");
}

// Alpha-blend any color (hex OR css var()) with transparent — safe for tokens.
export function mix(color: string, pct: number) {
  return `color-mix(in srgb, ${color} ${pct}%, transparent)`;
}

// ── Shared chart theme — CSS-var driven so SVG recolors live on theme toggle ──
export const CHART = {
  grid: "var(--chart-grid)",
  axis: "var(--chart-axis)",
  tickFont: 10,
  temp: "var(--chart-temp)",
  forecast: "var(--chart-forecast)",
  good: "var(--chart-good)",
  warn: "var(--chart-warn)",
  bad: "var(--chart-bad)",
  crit: "var(--chart-crit)",
  accent: "var(--chart-accent)",
  teal: "var(--chart-accent)",   // alias kept for back-compat
  cyan: "var(--chart-cyan)",
  indigo: "var(--chart-indigo)",
  amber: "var(--chart-amber)",
  fuchsia: "var(--chart-fuchsia)",
  combined: "var(--chart-combined)",
  radialTrack: "var(--chart-radial-track)",
};

export const tooltipStyle = {
  backgroundColor: "var(--chart-tooltip-bg)",
  border: "1px solid var(--chart-tooltip-border)",
  borderRadius: 12,
  fontSize: 12,
  color: "var(--text)",
  boxShadow: "var(--chart-tooltip-shadow)",
  padding: "8px 12px",
} as const;

export const tooltipLabelStyle = { color: "var(--text-muted)", fontSize: 11, marginBottom: 2 } as const;

// ── Health helpers (single source of truth across the app) ───────────────────
export function healthColor(score: number | null | undefined): string {
  if (score == null) return "var(--text-muted)";
  if (score >= 75) return "var(--good)";
  if (score >= 45) return "var(--warn)";
  return "var(--bad)";
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
          style={{ background: mix(accent, 12), border: `1px solid ${mix(accent, 30)}`, boxShadow: `0 6px 16px -8px ${mix(accent, 55)}` }}
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
  good: "text-emerald-700 border-emerald-200 bg-emerald-50 dark:text-emerald-300 dark:border-emerald-400/25 dark:bg-emerald-400/10",
  warn: "text-amber-700 border-amber-200 bg-amber-50 dark:text-amber-300 dark:border-amber-400/25 dark:bg-amber-400/10",
  bad: "text-rose-700 border-rose-200 bg-rose-50 dark:text-rose-300 dark:border-rose-400/25 dark:bg-rose-400/10",
  info: "text-sky-700 border-sky-200 bg-sky-50 dark:text-sky-300 dark:border-sky-400/25 dark:bg-sky-400/10",
  indigo: "text-indigo-700 border-indigo-200 bg-indigo-50 dark:text-indigo-300 dark:border-indigo-400/25 dark:bg-indigo-400/10",
  neutral: "text-slate-600 border-slate-200 bg-slate-50 dark:text-slate-300 dark:border-slate-500/25 dark:bg-slate-400/10",
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
  color = "var(--text)",
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
      {/* soft corner glow tinted to the metric color */}
      <span className="absolute -top-10 -right-10 w-32 h-32 rounded-full blur-2xl opacity-0 group-hover:opacity-30 transition-opacity duration-500" style={{ background: color }} />
      <span className="absolute left-0 top-5 bottom-5 w-[3px] rounded-full" style={{ background: color, boxShadow: `0 0 12px -1px ${mix(color, 70)}` }} />
      <div className="flex items-center justify-between pl-1.5">
        <span className="eyebrow">{label}</span>
        <span
          className="grid place-items-center w-9 h-9 rounded-xl shrink-0 transition-transform duration-300 group-hover:scale-110 group-hover:-rotate-3"
          style={{ background: mix(color, 12), border: `1px solid ${mix(color, 24)}`, boxShadow: `0 6px 18px -8px ${mix(color, 80)}` }}
        >
          <Icon className="w-4 h-4" style={{ color }} />
        </span>
      </div>
      <div className="flex items-end gap-1.5 pl-1.5">
        <span className="metric text-[34px] leading-none tracking-tight" style={{ color, filter: `drop-shadow(0 2px 10px ${mix(color, 28)})` }}>
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
          <RadialBar background={{ fill: "var(--chart-radial-track)" }} dataKey="value" cornerRadius={12} isAnimationActive={false} />
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
export function Sparkline({ data, color = "var(--accent)" }: { data: number[]; color?: string }) {
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
export function Progress({ value, color = "var(--accent)", track = "var(--surface-3)" }: { value: number; color?: string; track?: string }) {
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
      <span className="w-2 h-2 rounded-full pulse-dot" style={{ background: "var(--accent)", color: "var(--accent)" }} />
      <span className="text-sm">{label}</span>
    </div>
  );
}
