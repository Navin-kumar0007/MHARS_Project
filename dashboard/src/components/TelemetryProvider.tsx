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
  thresholds: {
    idle: number;
    safe_max: number;
    critical: number;
  };
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
};

type TelemetryContextType = {
  latest: Telemetry | null;
  history: Telemetry[];
  isConnected: boolean;
  systemStatus: SystemStatus | null;
  injectAnomaly: (type: string) => Promise<void>;
  switchMachine: (id: number) => Promise<void>;
  resetSystem: () => Promise<void>;
  refreshStatus: () => Promise<void>;
};

const API_BASE = "http://localhost:8050";
const WS_URL = "ws://localhost:8050/ws/telemetry";

const TelemetryContext = createContext<TelemetryContextType>({
  latest: null,
  history: [],
  isConnected: false,
  systemStatus: null,
  injectAnomaly: async () => {},
  switchMachine: async () => {},
  resetSystem: async () => {},
  refreshStatus: async () => {},
});

export const useTelemetry = () => useContext(TelemetryContext);

// ── Provider ──────────────────────────────────────────────────────────────────
export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const [latest, setLatest] = useState<Telemetry | null>(null);
  const [history, setHistory] = useState<Telemetry[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch system status
  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/system_status`);
      const data = await res.json();
      setSystemStatus(data);
    } catch (err) {
      console.error("Failed to fetch system status", err);
    }
  }, []);

  // Connect WebSocket
  useEffect(() => {
    refreshStatus();

    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        console.log("[WS] Connected");
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log("[WS] Disconnected — reconnecting in 2s...");
        setTimeout(connect, 2000);
      };

      ws.onerror = () => setIsConnected(false);

      ws.onmessage = (event) => {
        try {
          const payload: Telemetry = JSON.parse(event.data);
          setLatest(payload);
          setHistory((prev) => {
            const next = [...prev, payload];
            if (next.length > 120) next.shift(); // Keep 2 min of data
            return next;
          });
        } catch (err) {
          console.error("Failed to parse telemetry", err);
        }
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null; // Prevent reconnect on unmount
        wsRef.current.close();
      }
    };
  }, [refreshStatus]);

  // Actions
  const injectAnomaly = useCallback(async (type: string) => {
    await fetch(`${API_BASE}/api/inject_anomaly`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type }),
    });
  }, []);

  const switchMachine = useCallback(async (id: number) => {
    // Close existing WS — it will auto-reconnect
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
    }
    setHistory([]);
    setLatest(null);

    await fetch(`${API_BASE}/api/switch_machine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ machine_type_id: id }),
    });

    await refreshStatus();

    // Reconnect WS after a brief delay for backend to reinitialize
    setTimeout(() => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(() => {
          const retry = new WebSocket(WS_URL);
          wsRef.current = retry;
          retry.onopen = () => setIsConnected(true);
          retry.onclose = () => setIsConnected(false);
          retry.onmessage = (event) => {
            try {
              const payload: Telemetry = JSON.parse(event.data);
              setLatest(payload);
              setHistory((prev) => {
                const next = [...prev, payload];
                if (next.length > 120) next.shift();
                return next;
              });
            } catch (err) {
              console.error(err);
            }
          };
        }, 2000);
      };
      ws.onmessage = (event) => {
        try {
          const payload: Telemetry = JSON.parse(event.data);
          setLatest(payload);
          setHistory((prev) => {
            const next = [...prev, payload];
            if (next.length > 120) next.shift();
            return next;
          });
        } catch (err) {
          console.error(err);
        }
      };
    }, 1500);
  }, [refreshStatus]);

  const resetSystem = useCallback(async () => {
    await fetch(`${API_BASE}/api/reset`, { method: "POST" });
    setHistory([]);
    setLatest(null);
    await refreshStatus();
  }, [refreshStatus]);

  return (
    <TelemetryContext.Provider
      value={{
        latest,
        history,
        isConnected,
        systemStatus,
        injectAnomaly,
        switchMachine,
        resetSystem,
        refreshStatus,
      }}
    >
      {children}
    </TelemetryContext.Provider>
  );
}
