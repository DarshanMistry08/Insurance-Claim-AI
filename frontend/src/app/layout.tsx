import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Claims AI — Intelligent Insurance Processing",
  description: "Multi-agent AI pipeline for automated insurance claim validation using LangGraph and local LLM.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="bg-mesh min-h-screen">{children}</body>
    </html>
  );
}
