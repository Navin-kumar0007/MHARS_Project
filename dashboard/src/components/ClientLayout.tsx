"use client";

import React from "react";
import { TelemetryProvider } from "@/components/TelemetryProvider";
import Sidebar from "@/components/Sidebar";

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <TelemetryProvider>
      <div className="flex min-h-screen w-full bg-[#0a0a0e]">
        <Sidebar />
        <main className="flex-1 min-w-0 min-h-screen" style={{ paddingLeft: '256px' }}>{children}</main>
      </div>
    </TelemetryProvider>
  );
}
