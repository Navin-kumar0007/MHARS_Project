"use client";

import React from "react";
import { usePathname } from "next/navigation";
import { TelemetryProvider } from "@/components/TelemetryProvider";
import Sidebar from "@/components/Sidebar";

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  // Public / chrome-less routes: no sidebar, no telemetry padding.
  const bare = pathname === "/login" || pathname.startsWith("/shared");

  if (bare) {
    return (
      <TelemetryProvider>
        <main className="min-h-screen w-full">{children}</main>
      </TelemetryProvider>
    );
  }

  return (
    <TelemetryProvider>
      <div className="flex min-h-screen w-full">
        <Sidebar />
        <main className="flex-1 min-w-0 min-h-screen" style={{ paddingLeft: "256px" }}>
          {children}
        </main>
      </div>
    </TelemetryProvider>
  );
}
