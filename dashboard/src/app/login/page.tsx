"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useTelemetry } from "@/components/TelemetryProvider";
import { ChevronsLeftRightEllipsis, Lock, User, LogIn } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useTelemetry();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    const ok = await login(username, password);
    setBusy(false);
    if (ok) router.push("/");
    else setError("Invalid username or password.");
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-sm card card-accent p-8 fade-in">
        <div className="flex items-center gap-2.5 justify-center mb-1.5">
          <div className="grid place-items-center w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-indigo-600 border border-indigo-400/40 shadow-[0_6px_16px_-6px_rgba(94,106,210,0.7)]">
            <ChevronsLeftRightEllipsis className="w-6 h-6 text-white" />
          </div>
          <span className="text-2xl font-bold text-gradient tracking-tight">MHARS</span>
        </div>
        <p className="text-center text-[11px] text-[var(--text-dim)] tracking-wider mb-7">DIGITAL TWIN — SECURE ACCESS</p>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="text-xs text-[var(--text-dim)] mb-1.5 block">Username</label>
            <div className="flex items-center gap-2 bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-2.5 focus-within:border-indigo-400/50 transition-colors">
              <User className="w-4 h-4 text-[var(--text-dim)]" />
              <input value={username} onChange={(e) => setUsername(e.target.value)} className="bg-transparent flex-1 text-sm text-[var(--text)] outline-none" placeholder="admin" autoFocus />
            </div>
          </div>
          <div>
            <label className="text-xs text-[var(--text-dim)] mb-1.5 block">Password</label>
            <div className="flex items-center gap-2 bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-2.5 focus-within:border-indigo-400/50 transition-colors">
              <Lock className="w-4 h-4 text-[var(--text-dim)]" />
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="bg-transparent flex-1 text-sm text-[var(--text)] outline-none" placeholder="••••••••" />
            </div>
          </div>

          {error && <div className="text-xs text-rose-700 bg-rose-500/10 border border-rose-500/30 rounded-lg px-3 py-2">{error}</div>}

          <button type="submit" disabled={busy} className="btn-accent w-full flex items-center justify-center gap-2 font-semibold rounded-lg py-2.5 text-sm disabled:opacity-50">
            <LogIn className="w-4 h-4" /> {busy ? "Authenticating…" : "Sign In"}
          </button>
        </form>

        <div className="mt-6 pt-4 border-t border-[var(--border)] text-[10px] text-[var(--text-muted)] leading-relaxed">
          <p className="font-semibold text-[var(--text-dim)] mb-1">Demo credentials</p>
          <p>admin / admin123 · operator / oper123 · viewer / view123</p>
        </div>
      </div>
    </div>
  );
}
