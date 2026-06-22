"use client";

import React, { useEffect, useState, useCallback } from "react";
import { useTelemetry, ShareLink, AuthUser } from "@/components/TelemetryProvider";
import { Card, CardTitle, PageHeader, Badge } from "@/components/ui";
import {
  Settings as SettingsIcon,
  Cpu,
  Boxes,
  Share2,
  Users,
  Trash2,
  Plus,
  Copy,
  Check,
} from "lucide-react";

const inputCls =
  "bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-1.5 text-sm text-[var(--text)] outline-none focus:border-indigo-400/50 transition-colors";

export default function SettingsPage() {
  const {
    systemStatus, latest, can, isAuthenticated,
    createShareLink, listShareLinks, revokeShareLink,
    listUsers, createUser, updateUserRole, deleteUser,
  } = useTelemetry();

  const [links, setLinks] = useState<ShareLink[]>([]);
  const [label, setLabel] = useState("");
  const [hours, setHours] = useState(24);
  const [copied, setCopied] = useState<string | null>(null);
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [nu, setNu] = useState({ username: "", password: "", role: "viewer" });
  const [msg, setMsg] = useState("");

  const refreshLinks = useCallback(async () => { if (can("share")) setLinks(await listShareLinks()); }, [can, listShareLinks]);
  const refreshUsers = useCallback(async () => { if (can("manage_users")) setUsers(await listUsers()); }, [can, listUsers]);

  useEffect(() => { refreshLinks(); refreshUsers(); }, [refreshLinks, refreshUsers]);

  const onCreateLink = async () => {
    const tok = await createShareLink(label || "Shared status", hours);
    if (tok) { setLabel(""); await refreshLinks(); } else setMsg("Failed to create link (check permissions).");
  };
  const shareUrl = (tok: string) => `${typeof window !== "undefined" ? window.location.origin : ""}/shared/${tok}`;
  const copy = (tok: string) => { navigator.clipboard?.writeText(shareUrl(tok)); setCopied(tok); setTimeout(() => setCopied(null), 1500); };
  const onCreateUser = async () => {
    if (!nu.username || !nu.password) { setMsg("Username and password required."); return; }
    const ok = await createUser(nu.username, nu.password, nu.role);
    setMsg(ok ? "User created." : "Failed (already exists?).");
    setNu({ username: "", password: "", role: "viewer" });
    await refreshUsers();
  };

  const models = systemStatus?.models_loaded || {};
  const profile = systemStatus?.machine_profile;

  return (
    <div className="p-6 space-y-5 max-w-[1600px] mx-auto fade-in">
      <PageHeader icon={SettingsIcon} title="Settings" subtitle="Machine profile, model status, sharing and access control" />

      {msg && <div className="text-xs text-amber-700 bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2">{msg}</div>}

      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-6">
          <CardTitle icon={Cpu}>Machine Profile</CardTitle>
          {profile ? (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div><span className="eyebrow block">Name</span><span className="text-[var(--text)]">{profile.name}</span></div>
              <div><span className="eyebrow block">Idle</span><span className="text-emerald-700">{profile.idle}°C</span></div>
              <div><span className="eyebrow block">Safe Max</span><span className="text-amber-700">{profile.safe_max}°C</span></div>
              <div><span className="eyebrow block">Critical</span><span className="text-rose-700">{profile.critical}°C</span></div>
              <div><span className="eyebrow block">Heat Rate</span><span className="text-[var(--text)]">{profile.heat_rate}</span></div>
              <div><span className="eyebrow block">Mode</span><span className="text-[var(--text)]">{latest?.live_mode ? "Live" : "Demo"}</span></div>
            </div>
          ) : <p className="text-[var(--text-muted)] text-sm">Loading…</p>}
        </Card>

        <Card className="col-span-12 lg:col-span-6">
          <CardTitle icon={Boxes}>Model Status</CardTitle>
          {systemStatus?.model_status ? (
            <div className="grid grid-cols-1 gap-2">
              {Object.entries(systemStatus.model_status).map(([name, v]) => (
                <div key={name} className="flex items-center justify-between bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-2">
                  <span className="text-xs text-[var(--text)] truncate">{name}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] text-[var(--text-dim)]">{v.detail}</span>
                    <Badge tone={v.ok ? "good" : "warn"}>{v.ok ? "live" : "fallback"}</Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(models).map(([name, ok]) => (
                <div key={name} className="flex items-center justify-between bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-2">
                  <span className="text-xs text-[var(--text)] truncate">{name}</span>
                  <Badge tone={ok ? "good" : "neutral"}>{ok ? "loaded" : "fallback"}</Badge>
                </div>
              ))}
              {Object.keys(models).length === 0 && <p className="text-[var(--text-muted)] text-sm col-span-2">Loading…</p>}
            </div>
          )}
        </Card>

        {can("share") && (
          <Card className="col-span-12">
            <CardTitle icon={Share2}>Shareable Status Links</CardTitle>
            <div className="flex flex-wrap items-end gap-3 mb-4">
              <div>
                <label className="eyebrow block mb-1">Label</label>
                <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="For maintenance team" className={`${inputCls} w-56`} />
              </div>
              <div>
                <label className="eyebrow block mb-1">Expires (hours)</label>
                <input type="number" min={1} max={168} value={hours} onChange={(e) => setHours(Number(e.target.value))} className={`${inputCls} w-28`} />
              </div>
              <button onClick={onCreateLink} className="flex items-center gap-1.5 bg-indigo-400/15 hover:bg-indigo-400/25 border border-indigo-400/40 text-indigo-700 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors">
                <Plus className="w-4 h-4" /> Create Link
              </button>
            </div>
            <div className="space-y-2">
              {links.map((l) => (
                <div key={l.token} className="flex items-center justify-between bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-2 gap-3">
                  <div className="min-w-0">
                    <p className="text-sm text-[var(--text)] truncate">{l.label}</p>
                    <p className="text-[10px] text-[var(--text-dim)] truncate metric">{shareUrl(l.token)}</p>
                    <p className="text-[10px] text-[var(--text-muted)]">expires {new Date(l.expires_at * 1000).toLocaleString()} · {l.access_count} views</p>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button onClick={() => copy(l.token)} title="Copy URL" className="p-1.5 rounded-lg text-[var(--text-dim)] hover:text-indigo-600 hover:bg-indigo-400/10">
                      {copied === l.token ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <button onClick={async () => { await revokeShareLink(l.token); refreshLinks(); }} title="Revoke" className="p-1.5 rounded-lg text-[var(--text-dim)] hover:text-rose-400 hover:bg-rose-500/10">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
              {links.length === 0 && <p className="text-[var(--text-muted)] text-sm">No active share links.</p>}
            </div>
          </Card>
        )}

        {can("manage_users") && (
          <Card className="col-span-12">
            <CardTitle icon={Users}>User Management</CardTitle>
            <div className="flex flex-wrap items-end gap-3 mb-4">
              <div>
                <label className="eyebrow block mb-1">Username</label>
                <input value={nu.username} onChange={(e) => setNu({ ...nu, username: e.target.value })} className={`${inputCls} w-40`} />
              </div>
              <div>
                <label className="eyebrow block mb-1">Password</label>
                <input type="password" value={nu.password} onChange={(e) => setNu({ ...nu, password: e.target.value })} className={`${inputCls} w-40`} />
              </div>
              <div>
                <label className="eyebrow block mb-1">Role</label>
                <select value={nu.role} onChange={(e) => setNu({ ...nu, role: e.target.value })} className={inputCls}>
                  <option value="viewer">viewer</option>
                  <option value="operator">operator</option>
                  <option value="admin">admin</option>
                </select>
              </div>
              <button onClick={onCreateUser} className="flex items-center gap-1.5 bg-indigo-400/15 hover:bg-indigo-400/25 border border-indigo-400/40 text-indigo-700 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors">
                <Plus className="w-4 h-4" /> Add User
              </button>
            </div>
            <div className="space-y-2">
              {users.map((u) => (
                <div key={u.username} className="flex items-center justify-between bg-[var(--surface-3)] border border-[var(--border)] rounded-lg px-3 py-2 gap-3">
                  <span className="text-sm text-[var(--text)]">{u.username}</span>
                  <div className="flex items-center gap-2">
                    <select value={u.role} onChange={async (e) => { await updateUserRole(u.username, e.target.value); refreshUsers(); }} className={inputCls}>
                      <option value="viewer">viewer</option>
                      <option value="operator">operator</option>
                      <option value="admin">admin</option>
                    </select>
                    <button onClick={async () => { await deleteUser(u.username); refreshUsers(); }} title="Delete user" className="p-1.5 rounded-lg text-[var(--text-dim)] hover:text-rose-400 hover:bg-rose-500/10">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
              {users.length === 0 && <p className="text-[var(--text-muted)] text-sm">No users loaded.</p>}
            </div>
          </Card>
        )}

        {!isAuthenticated && (
          <div className="col-span-12 text-center text-[var(--text-muted)] text-sm py-6">
            Sign in to manage sharing and users. (Dev mode grants full access automatically.)
          </div>
        )}
      </div>
    </div>
  );
}
