"use client";

import Link from "next/link";
import { use, useEffect, useMemo, useRef, useState } from "react";

import type {
  Goalkeeper,
  Heatmap,
  MatchSummary,
  PassingNetwork,
  Player,
  Team,
  Timeline,
  TimelineEvent,
} from "@/lib/api";
import { api, formatClock, formatMetric } from "@/lib/api";
import {
  ConfidenceChip,
  CoverageRow,
  EmptyState,
  MetricCard,
  PitchHeatmap,
  Spinner,
  TeamChip,
  WarningPanel,
} from "@/components/ui";

type Tab = "overview" | "players" | "goalkeepers" | "teams" | "timeline" | "video";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "players", label: "Players" },
  { id: "goalkeepers", label: "Goalkeepers" },
  { id: "teams", label: "Teams" },
  { id: "timeline", label: "Timeline" },
  { id: "video", label: "Video" },
];

export default function MatchPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = use(params);
  const [tab, setTab] = useState<Tab>("overview");
  const [summary, setSummary] = useState<MatchSummary | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [teams, setTeams] = useState<Team[]>([]);
  const [keepers, setKeepers] = useState<Goalkeeper[]>([]);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [networks, setNetworks] = useState<PassingNetwork[]>([]);
  const [heatmaps, setHeatmaps] = useState<Heatmap[]>([]);
  const [error, setError] = useState("");

  // The video element is lifted to the page so the timeline can seek it from
  // any tab without the player unmounting and losing its position.
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    (async () => {
      try {
        const [s, p, t, g, tl, n, h] = await Promise.all([
          api.summary(jobId),
          api.players(jobId),
          api.teams(jobId),
          api.goalkeepers(jobId),
          api.timeline(jobId),
          api.networks(jobId),
          api.heatmaps(jobId),
        ]);
        setSummary(s);
        setPlayers(p);
        setTeams(t);
        setKeepers(g);
        setTimeline(tl);
        setNetworks(n);
        setHeatmaps(h);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, [jobId]);

  function seek(seconds: number) {
    setTab("video");
    // Let the video tab mount before seeking, otherwise the ref is still null.
    requestAnimationFrame(() => {
      const video = videoRef.current;
      if (video) {
        video.currentTime = Math.max(0, seconds);
        void video.play().catch(() => undefined);
      }
    });
  }

  if (error) {
    return (
      <div className="rounded-lg border border-rose-800/60 bg-rose-950/40 p-4 text-sm text-rose-200">
        {error}
      </div>
    );
  }
  if (!summary || !timeline) return <Spinner label="Loading analytics" />;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Link href="/" className="text-xs text-slate-500 hover:text-slate-300">
            ← Projects
          </Link>
          <h1 className="mt-1">Match analysis</h1>
          <div className="mt-1 text-sm text-slate-400">
            {summary.video_id} · {Number(summary.context.frames)} frames ·{" "}
            {Number(summary.context.duration_s).toFixed(1)}s
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <a className="btn-ghost" href={api.csvUrl(jobId, "players")}>
            Players CSV
          </a>
          <a className="btn-ghost" href={api.csvUrl(jobId, "events")}>
            Events CSV
          </a>
          <a className="btn-ghost" href={api.downloadUrl(jobId, "events")}>
            Events Parquet
          </a>
          <a className="btn-ghost" href={api.videoUrl(jobId, "annotated")}>
            Annotated video
          </a>
        </div>
      </div>

      <WarningPanel warnings={summary.data_quality.warnings} />

      <div className="flex gap-1 overflow-x-auto border-b border-slate-800">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? "tab-active" : "tab-idle"}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <Overview summary={summary} teams={teams} networks={networks} heatmaps={heatmaps} />
      )}
      {tab === "players" && <Players jobId={jobId} players={players} onSeek={seek} />}
      {tab === "goalkeepers" && <Goalkeepers keepers={keepers} />}
      {tab === "teams" && <Teams teams={teams} networks={networks} heatmaps={heatmaps} />}
      {tab === "timeline" && <TimelineView timeline={timeline} onSeek={seek} />}
      <div className={tab === "video" ? "" : "hidden"}>
        <VideoTab jobId={jobId} timeline={timeline} videoRef={videoRef} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ overview */

function Overview({
  summary,
  teams,
  networks,
  heatmaps,
}: {
  summary: MatchSummary;
  teams: Team[];
  networks: PassingNetwork[];
  heatmaps: Heatmap[];
}) {
  const q = summary.data_quality;
  const counts = Object.entries(summary.event_counts).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card">
          <div className="label">Ball observed</div>
          <div className="stat mt-1">{q.ball_observed_pct}%</div>
          <div className="mt-2 text-[11px] text-slate-500">
            {q.ball_known_pct}% known including interpolation
          </div>
        </div>
        <div className="card">
          <div className="label">Possession determinable</div>
          <div className="stat mt-1">{q.possession_determinable_pct}%</div>
          <div className="mt-2 text-[11px] text-slate-500">
            shares below are of this subset
          </div>
        </div>
        <div className="card">
          <div className="label">Usable player rows</div>
          <div className="stat mt-1">{q.valid_player_row_pct}%</div>
          <div className="mt-2 text-[11px] text-slate-500">physical stats are lower bounds</div>
        </div>
        <div className="card">
          <div className="label">Players tracked</div>
          <div className="stat mt-1">{summary.n_players}</div>
          <div className="mt-2 text-[11px] text-slate-500">
            {q.tracks_without_team} without a team
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <section className="card">
          <h2>Possession</h2>
          <div className="mt-4 space-y-3">
            {Object.entries(summary.possession.teams ?? {}).map(([team, value]) => (
              <div key={team}>
                <div className="mb-1 flex justify-between text-sm">
                  <span className="flex items-center gap-2">
                    <TeamChip teamId={team} />
                  </span>
                  <span className="tabular-nums">
                    {(value.share_of_controlled * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={`h-full ${team === "A" ? "bg-sky-500" : "bg-fuchsia-500"}`}
                    style={{ width: `${value.share_of_controlled * 100}%` }}
                  />
                </div>
              </div>
            ))}
            <div className="pt-2 text-[11px] text-slate-500">
              Unknown for {(summary.possession.unknown_ratio * 100).toFixed(1)}% of analysed
              time — excluded from the shares above rather than assigned.
            </div>
          </div>
        </section>

        <section className="card">
          <h2>Events detected</h2>
          <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
            {counts.map(([type, n]) => (
              <div key={type} className="flex justify-between border-b border-slate-800/60 py-1">
                <span className="text-slate-300">{type.replace(/_/g, " ")}</span>
                <span className="tabular-nums text-slate-400">{n}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {teams.map((team) => (
          <section key={team.team_id} className="card">
            <div className="flex items-center justify-between">
              <h2>
                <TeamChip teamId={team.team_id} /> overview
              </h2>
              <span className="text-[11px] text-slate-500">
                attacking {team.attack_direction.replace(/_/g, " ")}
              </span>
            </div>
            <PitchHeatmap
              heatmap={heatmaps.find(
                (h) => h.team_id === team.team_id && h.track_id === null && h.kind === "position",
              )}
            />
            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <Stat label="Passes" value={formatMetric(team.metrics.pass_attempts, 0)} />
              <Stat label="Pass accuracy" value={formatMetric(team.metrics.pass_accuracy_pct)} />
              <Stat label="Interceptions" value={formatMetric(team.metrics.interceptions, 0)} />
              <Stat label="Turnovers" value={formatMetric(team.metrics.turnovers, 0)} />
            </div>
            <div className="mt-3 text-[11px] text-slate-500">
              {networks.find((n) => n.team_id === team.team_id && n.window === "full_match")
                ?.n_passes ?? 0}{" "}
              passes in the network
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-800/40 px-3 py-2">
      <div className="label">{label}</div>
      <div className="mt-0.5 font-semibold tabular-nums">{value}</div>
    </div>
  );
}

/* ------------------------------------------------------------------- players */

function Players({
  jobId,
  players,
  onSeek,
}: {
  jobId: string;
  players: Player[];
  onSeek: (s: number) => void;
}) {
  const [selected, setSelected] = useState<Player | null>(null);
  const [heatmaps, setHeatmaps] = useState<Heatmap[]>([]);
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!selected) return;
    void api.heatmaps(jobId, { track_id: selected.track_id }).then(setHeatmaps);
  }, [jobId, selected]);

  const filtered = useMemo(
    () =>
      players.filter((p) =>
        p.display_name.toLowerCase().includes(query.toLowerCase()),
      ),
    [players, query],
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[380px_1fr]">
      <section className="card max-h-[70vh] overflow-y-auto">
        <input
          className="input mb-3"
          placeholder="Search players…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="space-y-1">
          {filtered.map((player) => (
            <button
              key={player.track_id}
              onClick={() => setSelected(player)}
              className={`w-full rounded-lg px-3 py-2 text-left transition-colors ${
                selected?.track_id === player.track_id
                  ? "bg-emerald-500/15 ring-1 ring-emerald-500/40"
                  : "hover:bg-slate-800/60"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-sm">{player.display_name}</span>
                <TeamChip teamId={player.team_id} />
              </div>
              <div className="mt-1 flex items-center gap-3 text-[11px] text-slate-500">
                <span>{formatMetric(player.metrics.distance_m, 0)}</span>
                <span>{formatMetric(player.metrics.touches, 0)} touches</span>
                <span>{(player.coverage.tracking * 100).toFixed(0)}% tracked</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      <section className="space-y-5">
        {!selected && <EmptyState title="Select a player" hint="Choose from the list." />}
        {selected && (
          <>
            <div className="card">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2>{selected.display_name}</h2>
                  <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
                    <TeamChip teamId={selected.team_id} />
                    <span>{selected.role}</span>
                    <span>
                      {formatClock(selected.first_seen_s)}–{formatClock(selected.last_seen_s)}
                    </span>
                  </div>
                </div>
                <CoverageRow coverage={selected.coverage} />
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="Distance" metric={selected.metrics.distance_m} digits={0} />
              <MetricCard label="Top speed" metric={selected.metrics.top_speed_m_s} digits={2} />
              <MetricCard label="Sprints" metric={selected.metrics.sprints} digits={0} />
              <MetricCard label="Touches" metric={selected.metrics.touches} digits={0} />
              <MetricCard label="Passes" metric={selected.metrics.pass_attempts} digits={0} />
              <MetricCard label="Pass accuracy" metric={selected.metrics.pass_accuracy_pct} />
              <MetricCard label="Carries" metric={selected.metrics.carries} digits={0} />
              <MetricCard
                label="Possession time"
                metric={selected.metrics.possession_time_s}
                digits={1}
              />
              <MetricCard
                label="Interceptions"
                metric={selected.metrics.interceptions}
                digits={0}
              />
              <MetricCard label="Recoveries" metric={selected.metrics.recoveries} digits={0} />
              <MetricCard
                label="Possession lost"
                metric={selected.metrics.possession_lost}
                digits={0}
              />
              <MetricCard label="Minutes tracked" metric={selected.metrics.minutes_tracked} />
            </div>

            <div className="card">
              <h2 className="mb-3">Heatmaps</h2>
              <div className="grid gap-5 sm:grid-cols-2">
                {heatmaps.map((h) => (
                  <PitchHeatmap key={`${h.kind}-${h.track_id}`} heatmap={h} />
                ))}
                {!heatmaps.length && (
                  <div className="text-sm text-slate-500">No heatmaps for this player.</div>
                )}
              </div>
            </div>

            <div className="card">
              <h2 className="mb-3">Events ({selected.event_links.length})</h2>
              <div className="max-h-72 space-y-1 overflow-y-auto">
                {selected.event_links.map((link) => (
                  <button
                    key={link.event_id}
                    onClick={() => onSeek(link.timestamp_s)}
                    className="flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-sm hover:bg-slate-800/60"
                  >
                    <span className="text-slate-300">{link.event_type.replace(/_/g, " ")}</span>
                    <span className="tabular-nums text-xs text-emerald-400">
                      {formatClock(link.timestamp_s)}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

/* --------------------------------------------------------------- goalkeepers */

function Goalkeepers({ keepers }: { keepers: Goalkeeper[] }) {
  if (!keepers.length) {
    return (
      <EmptyState
        title="No goalkeeper identified in this footage"
        hint="Phase 1 assigns the goalkeeper role from appearance and penalty-area occupancy. On wide midfield footage neither keeper may appear."
      />
    );
  }
  return (
    <div className="space-y-6">
      {keepers.map((keeper) => (
        <section key={keeper.track_id} className="card space-y-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2>{keeper.display_name}</h2>
              <TeamChip teamId={keeper.team_id} />
            </div>
            <CoverageRow coverage={keeper.coverage} />
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Shots faced" metric={keeper.metrics.shots_faced} digits={0} />
            <MetricCard
              label="Save candidates"
              metric={keeper.metrics.save_candidates}
              digits={0}
            />
            <MetricCard label="Distributions" metric={keeper.metrics.distributions} digits={0} />
            <MetricCard
              label="Distribution accuracy"
              metric={keeper.metrics.distribution_accuracy_pct}
            />
          </div>
          <p className="text-[11px] text-amber-500/80">{keeper.note}</p>
        </section>
      ))}
    </div>
  );
}

/* --------------------------------------------------------------------- teams */

function Teams({
  teams,
  networks,
  heatmaps,
}: {
  teams: Team[];
  networks: PassingNetwork[];
  heatmaps: Heatmap[];
}) {
  return (
    <div className="space-y-8">
      {teams.map((team) => {
        const network = networks.find(
          (n) => n.team_id === team.team_id && n.window === "full_match",
        );
        return (
          <section key={team.team_id} className="space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="flex items-center gap-2">
                <TeamChip teamId={team.team_id} /> — {team.n_players} players
              </h2>
              <CoverageRow coverage={team.coverage} />
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricCard label="Possession" metric={team.metrics.possession_pct} />
              <MetricCard label="Passes" metric={team.metrics.pass_attempts} digits={0} />
              <MetricCard label="Pass accuracy" metric={team.metrics.pass_accuracy_pct} />
              <MetricCard
                label="Progressive passes"
                metric={team.metrics.progressive_passes}
                digits={0}
              />
              <MetricCard label="Distance" metric={team.metrics.distance_m} digits={0} />
              <MetricCard label="Sprints" metric={team.metrics.sprints} digits={0} />
              <MetricCard
                label="Final third entries"
                metric={team.metrics.final_third_entries}
                digits={0}
              />
              <MetricCard
                label="Penalty area entries"
                metric={team.metrics.penalty_area_entries}
                digits={0}
              />
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              <div className="card">
                <h2 className="mb-3">Average positions & passing network</h2>
                <PassingNetworkView network={network} />
              </div>
              <div className="card">
                <h2 className="mb-3">Team heatmaps</h2>
                <div className="grid gap-4 sm:grid-cols-2">
                  {heatmaps
                    .filter((h) => h.team_id === team.team_id && h.track_id === null)
                    .map((h) => (
                      <PitchHeatmap key={h.kind} heatmap={h} height={150} />
                    ))}
                </div>
              </div>
            </div>
          </section>
        );
      })}
    </div>
  );
}

function PassingNetworkView({ network }: { network?: PassingNetwork }) {
  if (!network || !network.nodes.length) {
    return <div className="text-sm text-slate-500">No passing network available.</div>;
  }
  const positioned = network.nodes.filter((n) => n.x !== null && n.y !== null);
  const maxCount = Math.max(1, ...network.edges.map((e) => e.count));

  return (
    <div className="space-y-3">
      <svg viewBox="0 0 105 68" className="w-full rounded-lg">
        <rect width="105" height="68" fill="#14622f" />
        <g stroke="#e8f5ec" strokeWidth="0.35" fill="none" opacity="0.7">
          <rect x="0.2" y="0.2" width="104.6" height="67.6" />
          <line x1="52.5" y1="0" x2="52.5" y2="68" />
          <circle cx="52.5" cy="34" r="9.15" />
        </g>
        {network.edges.map((edge) => {
          const a = positioned.find((n) => n.track_id === edge.source);
          const b = positioned.find((n) => n.track_id === edge.target);
          if (!a || !b) return null;
          return (
            <line
              key={`${edge.source}-${edge.target}`}
              x1={a.x!}
              y1={68 - a.y!}
              x2={b.x!}
              y2={68 - b.y!}
              stroke="#fbbf24"
              strokeWidth={0.3 + (edge.count / maxCount) * 1.2}
              opacity={0.75}
            />
          );
        })}
        {positioned.map((node) => (
          <g key={node.track_id}>
            <circle
              cx={node.x!}
              cy={68 - node.y!}
              r={1.4 + node.centrality * 2.2}
              fill="#0ea5e9"
              stroke="#0f172a"
              strokeWidth="0.3"
            />
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap justify-between gap-2 text-[11px] text-slate-500">
        <span>{network.n_passes} passes · dominant side: {network.dominant_side}</span>
        <span>
          {network.isolated_players.length} isolated ·{" "}
          {network.most_influential !== null ? `most involved #${network.most_influential}` : "—"}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ timeline */

function TimelineView({
  timeline,
  onSeek,
}: {
  timeline: Timeline;
  onSeek: (s: number) => void;
}) {
  const [type, setType] = useState("");
  const [team, setTeam] = useState("");
  const [band, setBand] = useState("");

  const events = useMemo(
    () =>
      timeline.events.filter(
        (e) =>
          (!type || e.type === type) &&
          (!team || e.team_id === team) &&
          (!band || e.confidence_band === band),
      ),
    [timeline.events, type, team, band],
  );

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap gap-3">
        <select className="input w-52" value={type} onChange={(e) => setType(e.target.value)}>
          <option value="">All event types</option>
          {timeline.filters.types.map((t) => (
            <option key={t} value={t}>
              {t.replace(/_/g, " ")}
            </option>
          ))}
        </select>
        <select className="input w-40" value={team} onChange={(e) => setTeam(e.target.value)}>
          <option value="">All teams</option>
          {timeline.filters.teams.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <select className="input w-44" value={band} onChange={(e) => setBand(e.target.value)}>
          <option value="">Any confidence</option>
          {timeline.filters.confidence_bands.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        <div className="ml-auto self-center text-sm text-slate-400">
          {events.length} of {timeline.events.length} events
        </div>
      </div>

      <div className="card">
        <div className="relative h-12 overflow-hidden rounded-lg bg-slate-800/50">
          {events.map((event) => (
            <button
              key={event.event_id}
              title={`${event.type} — ${formatClock(event.timestamp_s)}`}
              onClick={() => onSeek(event.timestamp_s)}
              className="absolute top-0 h-full w-[3px] -translate-x-1/2 bg-emerald-400/70 hover:w-[5px] hover:bg-emerald-300"
              style={{ left: `${(event.timestamp_s / timeline.duration_s) * 100}%` }}
            />
          ))}
        </div>
        <div className="mt-1 flex justify-between text-[11px] text-slate-500">
          <span>00:00</span>
          <span>{formatClock(timeline.duration_s)}</span>
        </div>
      </div>

      <div className="card max-h-[55vh] overflow-y-auto p-0">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-slate-900 text-left text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-4 py-2.5">Time</th>
              <th className="px-4 py-2.5">Event</th>
              <th className="px-4 py-2.5">Player</th>
              <th className="px-4 py-2.5">Team</th>
              <th className="px-4 py-2.5">Confidence</th>
              <th className="px-4 py-2.5">Ball</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {events.map((event: TimelineEvent) => (
              <tr
                key={event.event_id}
                onClick={() => onSeek(event.timestamp_s)}
                className="cursor-pointer hover:bg-slate-800/40"
              >
                <td className="px-4 py-2 tabular-nums text-emerald-400">
                  {formatClock(event.timestamp_s)}
                </td>
                <td className="px-4 py-2">{event.type.replace(/_/g, " ")}</td>
                <td className="px-4 py-2 text-slate-300">{event.player_name || "—"}</td>
                <td className="px-4 py-2">
                  {event.team_id ? <TeamChip teamId={event.team_id} /> : "—"}
                </td>
                <td className="px-4 py-2">
                  <ConfidenceChip band={event.confidence_band} />
                </td>
                <td className="px-4 py-2 text-xs text-slate-500">{event.ball_state}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!events.length && (
          <div className="p-6 text-center text-sm text-slate-500">
            No events match these filters.
          </div>
        )}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- video */

function VideoTab({
  jobId,
  timeline,
  videoRef,
}: {
  jobId: string;
  timeline: Timeline;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}) {
  const [kind, setKind] = useState("annotated");
  const [current, setCurrent] = useState(0);

  const nearby = useMemo(
    () => timeline.events.filter((e) => Math.abs(e.timestamp_s - current) < 2.5).slice(0, 6),
    [timeline.events, current],
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
      <div className="space-y-3">
        <div className="flex flex-wrap gap-2">
          {["annotated", "radar", "combined"].map((k) => (
            <button
              key={k}
              className={k === kind ? "btn-primary" : "btn-ghost"}
              onClick={() => setKind(k)}
            >
              {k}
            </button>
          ))}
          <a className="btn-ghost ml-auto" href={api.videoUrl(jobId, kind)} download>
            Download
          </a>
        </div>
        <video
          ref={videoRef}
          key={kind}
          src={api.videoUrl(jobId, kind)}
          controls
          className="w-full rounded-xl border border-slate-800 bg-black"
          onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        />
        <div className="relative h-8 overflow-hidden rounded-lg bg-slate-800/50">
          {timeline.events.map((event) => (
            <button
              key={event.event_id}
              title={`${event.type} — ${formatClock(event.timestamp_s)}`}
              onClick={() => {
                if (videoRef.current) videoRef.current.currentTime = event.timestamp_s;
              }}
              className="absolute top-0 h-full w-[3px] -translate-x-1/2 bg-emerald-400/60 hover:bg-emerald-300"
              style={{ left: `${(event.timestamp_s / timeline.duration_s) * 100}%` }}
            />
          ))}
          <div
            className="absolute top-0 h-full w-0.5 bg-white"
            style={{ left: `${(current / timeline.duration_s) * 100}%` }}
          />
        </div>
      </div>

      <aside className="card h-fit">
        <h2 className="mb-3">At {formatClock(current)}</h2>
        {nearby.length === 0 && (
          <div className="text-sm text-slate-500">No events near this moment.</div>
        )}
        <div className="space-y-2">
          {nearby.map((event) => (
            <div key={event.event_id} className="rounded-lg bg-slate-800/40 px-3 py-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">{event.type.replace(/_/g, " ")}</span>
                <ConfidenceChip band={event.confidence_band} />
              </div>
              <div className="mt-1 text-[11px] text-slate-400">
                {event.player_name || "—"}
                {event.related_player_name && ` → ${event.related_player_name}`}
              </div>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
