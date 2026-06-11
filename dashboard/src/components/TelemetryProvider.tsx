"use client";

import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from "react";

// ── Types ─────────────────────────────────────────────────────────────────────
export type Telemetry = {
  timestamp: number;
  machine_type: string;
  machine_type_id: number;
  current_temp: number;
  lstm_prediction: number;
  if_score: number;
  lstm_score: number;
  ae_score: number;
  vib_score: number;
  anomaly_score: number;
  context_score: number;
  urgency: number;
  action: string;
  route: string;
  latency_ms: number;
  alert: string;
  llm_source: string;
  raw_obs: number[];
  active_anomaly: string | null;
  anomaly_ticks_remaining: number;
  live_mode: boolean;
  thresholds: {
    idle: number;
    safe_max: number;
    critical: number;
  };
  system_health?: {
    title: string;
    overall_score: number;
    ai_summary: string;
    components: { icon: string; name: string; val: string; verdict: string; status: string }[];
  };
  metadata: any;
};

export type SystemStatus = {
  machine_type_id: number;
  machine_profile: {
    name: string;
    safe_max: number;
    critical: number;
    idle: number;
    load: number;
    heat_rate: number;
  };
  models_loaded: Record<string, boolean>;
  available_machines: Record<string, string>;
  available_anomalies: Record<string, string>;
  active_anomaly: string | null;
  synthetic_mode?: boolean;
};

export type AuthUser = { username: string; role: string };

export type ShareLink = {
  token: string;
  label: string;
  creator: string;
  created_at: number;
  expires_at: number;
  access_count: number;
};

type TelemetryContextType = {
  latest: Telemetry | null;
  history: Telemetry[];
  isConnected: boolean;
  systemStatus: SystemStatus | null;
  registry: Record<string, any>;
  liveMode: boolean;
  // Auth
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  authReady: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  can: (action: string) => boolean;
  // Actions
  injectAnomaly: (type: string) => Promise<void>;
  switchMachine: (id: number) => Promise<void>;
  resetSystem: () => Promise<void>;
  toggleMode: () => Promise<void>;
  refreshStatus: () => Promise<void>;
  // Enterprise features
  createShareLink: (label: string, expiresInHours: number) => Promise<string | null>;
  listShareLinks: () => Promise<ShareLink[]>;
  revokeShareLink: (token: string) => Promise<boolean>;
  downloadReport: () => Promise<void>;
  listUsers: () => Promise<AuthUser[]>;
  createUser: (username: string, password: string, role: string) => Promise<boolean>;
  updateUserRole: (username: string, role: string) => Promise<boolean>;
  deleteUser: (username: string) => Promise<boolean>;
};

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/telemetry";

// Role → permitted actions (mirrors backend RBAC matrix)
const ROLE_PERMS: Record<string, string[]> = {
  viewer: ["view", "download_report"],
  operator: ["view", "download_report", "inject", "switch", "share", "toggle_mode"],
  admin: ["view", "download_report", "inject", "switch", "share", "toggle_mode", "manage_users"],
};

const noop = async () => {};

const TelemetryContext = createContext<TelemetryContextType>({
  latest: null,
  history: [],
  isConnected: false,
  systemStatus: null,
  registry: {},
  liveMode: false,
  user: null,
  token: null,
  isAuthenticated: false,
  authReady: false,
  login: async () => false,
  logout: () => {},
  can: () => false,
  injectAnomaly: noop,
  switchMachine: noop,
  resetSystem: noop,
  toggleMode: noop,
  refreshStatus: noop,
  createShareLink: async () => null,
  listShareLinks: async () => [],
  revokeShareLink: async () => false,
  downloadReport: async () => {},
  listUsers: async () => [],
  createUser: async () => false,
  updateUserRole: async () => false,
  deleteUser: async () => false,
});

export const useTelemetry = () => useContext(TelemetryContext);

// ── Provider ──────────────────────────────────────────────────────────────────
export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const [latest, setLatest] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [registry, setRegistry] = useState<Record<string, any>>({});
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const tokenRef = useRef<string | null>(null);

  const liveMode = latest?.live_mode ?? false;

  // Build auth headers from the current token (if any)
  const authHeaders = useCallback((): Record<string, string> => {
    return tokenRef.current ? { Authorization: `Bearer ${tokenRef.current}` } : {};
  }, []);

  // ── Auth bootstrap: restore token from localStorage, verify with /me ──────────
  useEffect(() => {
    const stored = typeof window !== "undefined" ? localStorage.getItem("mhars_token") : null;
    const finish = () => setAuthReady(true);
    if (stored) {
      tokenRef.current = stored;
      setToken(stored);
      fetch(`${API_BASE}/api/auth/me`, { headers: { Authorization: `Bearer ${stored}` } })
        .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
        .then((u) => setUser(u))
        .catch(() => {
          // invalid/expired token — clear it
          localStorage.removeItem("mhars_token");
          tokenRef.current = null;
          setToken(null);
        })
        .finally(finish);
    } else {
      // Dev mode: backend may not require auth — probe /me without a token
      fetch(`${API_BASE}/api/auth/me`)
        .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
        .then((u) => setUser(u))
        .catch(() => setUser(null))
        .finally(finish);
    }
  }, []);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    try {
      const body = new URLSearchParams({ username, password });
      const res = await fetch(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
      });
      if (!res.ok) return false;
      const data = await res.json();
      const tok = data.access_token as string;
      tokenRef.current = tok;
      setToken(tok);
      if (typeof window !== "undefined") localStorage.setItem("mhars_token", tok);
      setUser({ username, role: data.role });
      return true;
    } catch {
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    tokenRef.current = null;
    setToken(null);
    setUser(null);
    if (typeof window !== "undefined") localStorage.removeItem("mhars_token");
  }, []);

  const can = useCallback(
    (action: string) => {
      const role = user?.role || "viewer";
      return (ROLE_PERMS[role] || []).includes(action);
    },
    [user]
  );

  // ── Data fetchers ─────────────────────────────────────────────────────────────
  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/system_status`);
      setSystemStatus(await res.json());
    } catch (err) {
      console.error("Failed to fetch system status", err);
    }
  }, []);

  const refreshRegistry = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/registry`);
      setRegistry(await res.json());
    } catch (err) {
      console.error("Failed to fetch registry", err);
    }
  }, []);

  // ── WebSocket stream (token appended when available) ─────────────────────────
  const pushTelemetry = useCallback((payload: Telemetry) => {
    setLatest(payload);
    setHistory((prev) => {
      const next = [...prev, payload];
      if (next.length > 120) next.shift();
      return next;
    });
  }, []);

  useEffect(() => {
    if (!authReady) return;
    refreshStatus();
    refreshRegistry();
    const registryInterval = setInterval(refreshRegistry, 5000);

    let stopped = false;
    const connect = () => {
      if (stopped) return;
      const url = tokenRef.current ? `${WS_BASE}?token=${encodeURIComponent(tokenRef.current)}` : WS_BASE;
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        if (!stopped) setTimeout(connect, 2000);
      };
      ws.onerror = () => setIsConnected(false);
      ws.onmessage = (event) => {
        try {
          pushTelemetry(JSON.parse(event.data));
        } catch (err) {
          console.error("Failed to parse telemetry", err);
        }
      };
    };
    connect();

    return () => {
      stopped = true;
      clearInterval(registryInterval);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [authReady, token, refreshStatus, refreshRegistry, pushTelemetry]);

  // ── Mutations ─────────────────────────────────────────────────────────────────
  const injectAnomaly = useCallback(async (type: string) => {
    await fetch(`${API_BASE}/api/inject_anomaly`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ type }),
    });
  }, [authHeaders]);

  const switchMachine = useCallback(async (id: number) => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }
    setHistory([]);
    setLatest(null);
    await fetch(`${API_BASE}/api/switch_machine`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ machine_type_id: id }),
    });
    await refreshStatus();
    // WS auto-reconnects via the effect's close handler.
    const url = tokenRef.current ? `${WS_BASE}?token=${encodeURIComponent(tokenRef.current)}` : WS_BASE;
    setTimeout(() => {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(() => {
          const retry = new WebSocket(url);
          wsRef.current = retry;
          retry.onopen = () => setIsConnected(true);
          retry.onclose = () => setIsConnected(false);
          retry.onmessage = (e) => { try { pushTelemetry(JSON.parse(e.data)); } catch {} };
        }, 2000);
      };
      ws.onmessage = (e) => { try { pushTelemetry(JSON.parse(e.data)); } catch {} };
    }, 1500);
  }, [authHeaders, refreshStatus, pushTelemetry]);

  const resetSystem = useCallback(async () => {
    await fetch(`${API_BASE}/api/reset`, { method: "POST", headers: { ...authHeaders() } });
    setHistory([]);
    setLatest(null);
    await refreshStatus();
  }, [authHeaders, refreshStatus]);

  const toggleMode = useCallback(async () => {
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }
    setHistory([]);
    setLatest(null);
    await fetch(`${API_BASE}/api/toggle_mode`, { method: "POST", headers: { ...authHeaders() } });
    await refreshStatus();
    const url = tokenRef.current ? `${WS_BASE}?token=${encodeURIComponent(tokenRef.current)}` : WS_BASE;
    setTimeout(() => {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => setIsConnected(false);
      ws.onmessage = (e) => { try { pushTelemetry(JSON.parse(e.data)); } catch {} };
    }, 1500);
  }, [authHeaders, refreshStatus, pushTelemetry]);

  // ── Enterprise features ─────────────────────────────────────────────────────
  const createShareLink = useCallback(async (label: string, expiresInHours: number): Promise<string | null> => {
    try {
      const res = await fetch(`${API_BASE}/api/share/create`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ label, expires_in_hours: expiresInHours }),
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data.token as string;
    } catch {
      return null;
    }
  }, [authHeaders]);

  const listShareLinks = useCallback(async (): Promise<ShareLink[]> => {
    try {
      const res = await fetch(`${API_BASE}/api/share/list`, { headers: { ...authHeaders() } });
      if (!res.ok) return [];
      const data = await res.json();
      return (data.links || []) as ShareLink[];
    } catch {
      return [];
    }
  }, [authHeaders]);

  const revokeShareLink = useCallback(async (tok: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/share/revoke/${tok}`, {
        method: "POST",
        headers: { ...authHeaders() },
      });
      return res.ok;
    } catch {
      return false;
    }
  }, [authHeaders]);

  const downloadReport = useCallback(async () => {
    // Fetch with auth header, then open the HTML report in a new tab
    // (user can print → save as PDF). Falls back to plain open in dev mode.
    try {
      const res = await fetch(`${API_BASE}/api/report/html`, { headers: { ...authHeaders() } });
      if (!res.ok) throw new Error(String(res.status));
      const html = await res.text();
      const blob = new Blob([html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch {
      window.open(`${API_BASE}/api/report/html`, "_blank");
    }
  }, [authHeaders]);

  const listUsers = useCallback(async (): Promise<AuthUser[]> => {
    try {
      const res = await fetch(`${API_BASE}/api/users`, { headers: { ...authHeaders() } });
      if (!res.ok) return [];
      const data = await res.json();
      return (data.users || data || []) as AuthUser[];
    } catch {
      return [];
    }
  }, [authHeaders]);

  const createUser = useCallback(async (username: string, password: string, role: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ username, password, role }),
      });
      return res.ok;
    } catch {
      return false;
    }
  }, [authHeaders]);

  const updateUserRole = useCallback(async (username: string, role: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/users/${username}/role`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ role }),
      });
      return res.ok;
    } catch {
      return false;
    }
  }, [authHeaders]);

  const deleteUser = useCallback(async (username: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/api/users/${username}`, {
        method: "DELETE",
        headers: { ...authHeaders() },
      });
      return res.ok;
    } catch {
      return false;
    }
  }, [authHeaders]);

  return (
    <TelemetryContext.Provider
      value={{
        latest,
        history,
        isConnected,
        systemStatus,
        registry,
        liveMode,
        user,
        token,
        isAuthenticated: !!user,
        authReady,
        login,
        logout,
        can,
        injectAnomaly,
        switchMachine,
        resetSystem,
        toggleMode,
        refreshStatus,
        createShareLink,
        listShareLinks,
        revokeShareLink,
        downloadReport,
        listUsers,
        createUser,
        updateUserRole,
        deleteUser,
      }}
    >
      {children}
    </TelemetryContext.Provider>
  );
}
