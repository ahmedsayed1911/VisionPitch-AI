"use client";

/**
 * Shared presentation components.
 *
 * The recurring theme: a number is never shown without the coverage behind it.
 * `MetricCard` renders an em dash rather than a zero when a metric is not
 * reportable, because "no data" and "zero" are different facts and a dashboard
 * that conflates them teaches the user to trust numbers that are not there.
 */

import type { Coverage, Heatmap, Metric } from "@/lib/api";
import { coverageTone, formatMetric } from "@/lib/api";

export function CoverageBadge({ value, label }: { value: number; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5" title={`${label} coverage`}>
      <span className="label">{label}</span>
      <span className={`text-xs font-semibold tabular-nums ${coverageTone(value)}`}>
        {(value * 100).toFixed(0)}%
      </span>
    </span>
  );
}

export function CoverageRow({ coverage }: { coverage: Coverage }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      <CoverageBadge value={coverage.tracking} label="tracking" />
      <CoverageBadge value={coverage.pitch} label="pitch" />
      <CoverageBadge value={coverage.ball} label="ball" />
      <CoverageBadge value={coverage.identity} label="identity" />
    </div>
  );
}

export function MetricCard({
  label,
  metric,
  digits = 1,
}: {
  label: string;
  metric?: Metric;
  digits?: number;
}) {
  const reportable = metric?.reportable ?? false;
  return (
    <div className="card">
      <div className="label">{label}</div>
      <div className={`stat mt-1 ${reportable ? "" : "text-slate-600"}`}>
        {formatMetric(metric, digits)}
      </div>
      {metric && (
        <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
          <span className={coverageTone(metric.coverage)}>
            {(metric.coverage * 100).toFixed(0)}% coverage
          </span>
          <span title={`basis: ${metric.basis}`}>n={metric.n_samples}</span>
        </div>
      )}
      {metric && metric.basis === "includes_extrapolated" && (
        <div className="mt-1 text-[10px] text-amber-500/80">includes extrapolated positions</div>
      )}
    </div>
  );
}

export function ConfidenceChip({ band }: { band: string }) {
  const tone =
    band === "high"
      ? "bg-emerald-500/15 text-emerald-300"
      : band === "probable"
        ? "bg-amber-500/15 text-amber-300"
        : "bg-rose-500/15 text-rose-300";
  return <span className={`chip ${tone}`}>{band}</span>;
}

export function TeamChip({ teamId }: { teamId: string }) {
  const tone =
    teamId === "A"
      ? "bg-sky-500/15 text-sky-300"
      : teamId === "B"
        ? "bg-fuchsia-500/15 text-fuchsia-300"
        : "bg-slate-600/20 text-slate-400";
  return <span className={`chip ${tone}`}>{teamId === "none" ? "official" : `Team ${teamId}`}</span>;
}

export function WarningPanel({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <div className="rounded-xl border border-amber-800/50 bg-amber-950/30 p-4">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-amber-300">
        Data quality — read before quoting these numbers
      </div>
      <ul className="space-y-1.5 text-sm text-amber-100/80">
        {warnings.map((w) => (
          <li key={w} className="flex gap-2">
            <span className="text-amber-500">•</span>
            <span>{w}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** A pitch drawn to scale with a heatmap overlaid. */
export function PitchHeatmap({
  heatmap,
  height = 220,
}: {
  heatmap?: Heatmap;
  height?: number;
}) {
  const width = Math.round((height * 105) / 68);
  const cells = heatmap?.grid ?? [];
  const max = cells.length ? Math.max(...cells.flat()) : 0;

  return (
    <div className="space-y-2">
      <svg
        viewBox="0 0 105 68"
        style={{ width: "100%", maxWidth: width * 1.6, height: "auto" }}
        className="rounded-lg"
      >
        <rect x="0" y="0" width="105" height="68" fill="#14622f" />
        {cells.map((row, y) =>
          row.map((v, x) => {
            if (!max || v <= 0) return null;
            const w = 105 / (heatmap?.grid_x ?? 12);
            const h = 68 / (heatmap?.grid_y ?? 8);
            return (
              <rect
                key={`${x}-${y}`}
                x={x * w}
                y={68 - (y + 1) * h}
                width={w}
                height={h}
                fill="#fbbf24"
                opacity={Math.min(0.85, (v / max) * 0.85)}
              />
            );
          }),
        )}
        <g stroke="#e8f5ec" strokeWidth="0.4" fill="none" opacity="0.85">
          <rect x="0.2" y="0.2" width="104.6" height="67.6" />
          <line x1="52.5" y1="0" x2="52.5" y2="68" />
          <circle cx="52.5" cy="34" r="9.15" />
          <rect x="0.2" y="13.84" width="16.5" height="40.32" />
          <rect x="88.3" y="13.84" width="16.5" height="40.32" />
          <rect x="0.2" y="24.84" width="5.5" height="18.32" />
          <rect x="99.3" y="24.84" width="5.5" height="18.32" />
        </g>
      </svg>
      {heatmap && (
        <div className="flex items-center justify-between text-[11px] text-slate-500">
          <span>{heatmap.kind.replace(/_/g, " ")}</span>
          <span className={heatmap.reportable ? "" : "text-rose-400"}>
            {heatmap.n_samples} samples
            {!heatmap.reportable && " — too few to read"}
          </span>
        </div>
      )}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-10 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-emerald-400" />
      {label}…
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="card text-center">
      <div className="text-slate-300">{title}</div>
      {hint && <div className="mt-1 text-sm text-slate-500">{hint}</div>}
    </div>
  );
}
