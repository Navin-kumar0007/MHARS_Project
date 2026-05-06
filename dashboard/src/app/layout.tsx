import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import ClientLayout from "@/components/ClientLayout";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MHARS Digital Twin — Real-Time AI Monitoring Dashboard",
  description:
    "Multi-modal Hybrid Adaptive Response System: Interactive machine health monitoring with live AI inference, anomaly detection, and RL-based response visualization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="min-h-full bg-[#0A0A0E] text-slate-200">
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
