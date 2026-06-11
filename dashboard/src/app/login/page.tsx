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
          <div className="grid place-items-center w-10 h-10 rounded-xl bg-gradient-to-br from-teal-400/20 to-indigo-500/20 border border-teal-400/30">
            <ChevronsLeftRightEllipsis className="w-6 h-6 text-teal-300" />
          </div>
          <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-300 to-indigo-300 tracking-tight">MHARS</span>
        </div>
        <p className="text-center text-[11px] text-slate-500 tracking-wider mb-7">DIGITAL TWIN — SECURE ACCESS</p>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="text-xs text-slate-400 mb-1.5 block">Username</label>
            <div className="flex items-center gap-2 bg-white/[0.03] border border-white/[0.08] rounded-lg px-3 py-2.5 focus-within:border-teal-400/50 transition-colors">
              <User className="w-4 h-4 text-slate-500" />
              <input value={username} onChange={(e) => setUsername(e.target.value)} className="bg-transparent flex-1 text-sm text-slate-200 outline-none" placeholder="admin" autoFocus />
            </div>
          </div>
          <div>
            <label className="text-xs text-slate-400 mb-1.5 block">Password</label>
            <div className="flex items-center gap-2 bg-white/[0.03] border border-white/[0.08] rounded-lg px-3 py-2.5 focus-within:border-teal-400/50 transition-colors">
              <Lock className="w-4 h-4 text-slate-500" />
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="bg-transparent flex-1 text-sm text-slate-200 outline-none" placeholder="••••••••" />
            </div>
          </div>

          {error && <div className="text-xs text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg px-3 py-2">{error}</div>}

          <button type="submit" disabled={busy} className="w-full flex items-center justify-center gap-2 bg-teal-400/15 hover:bg-teal-400/25 border border-teal-400/40 text-teal-200 font-semibold rounded-lg py-2.5 text-sm transition-colors disabled:opacity-50">
            <LogIn className="w-4 h-4" /> {busy ? "Authenticating…" : "Sign In"}
          </button>
        </form>

        <div className="mt-6 pt-4 border-t border-white/[0.06] text-[10px] text-slate-600 leading-relaxed">
          <p className="font-semibold text-slate-500 mb-1">Demo credentials</p>
          <p>admin / admin123 · operator / oper123 · viewer / view123</p>
        </div>
      </div>
    </div>
  );
}
