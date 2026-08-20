import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "VisionPitch AI",
  description: "Football video analysis — players, teams, events and heatmaps.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/85 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-3.5">
            <Link href="/" className="flex items-center gap-2.5">
              <span className="grid h-7 w-7 place-items-center rounded-md bg-emerald-500 text-sm font-bold text-slate-950">
                V
              </span>
              <span className="text-[15px] font-semibold tracking-tight">VisionPitch AI</span>
              <span className="ml-1 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
                Phase 2
              </span>
            </Link>
            <nav className="flex items-center gap-1 text-sm">
              <Link href="/" className="rounded-md px-3 py-1.5 text-slate-300 hover:bg-slate-800">
                Projects
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-[1400px] px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
