/**
 * Typed client for the VisionPitch API.
 *
 * Every analytics value arrives as a Metric carrying its coverage, and the UI
 * types mirror that: there is no way to render a number here without having its
 * coverage in scope, which is the point.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export type MetricBasis =
  | "valid_only"
  | "includes_extrapolated"
  | "event_derived"
  | "image_space";

export interface Metric {
  value: number | null;
  coverage: number;
  confidence: number;
  n_samples: number;
  basis: MetricBasis;
  unit: string;
  reportable: boolean;
}

export interface Coverage {
  tracking: number;
  pitch: number;
  ball: number;
  identity: number;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  n_jobs: number;
  jobs: Job[];
}

export interface Job {
  id: string;
  project_id: string;
  name: string;
  mode: string;
  status:
    | "pending"
    | "running"
    | "analysing"
    | "completed"
    | "failed"
    | "cancelled";
  progress: number;
  stage: string;
  video_filename: string;
  created_at: string;
  finished_at: string | null;
  has_analytics: boolean;
  error: string;
  quality?: DataQuality;
}

export interface DataQuality {
  ball_known_pct: number;
  ball_observed_pct: number;
  valid_player_row_pct: number;
  possession_determinable_pct: number;
  tracks_without_team: number;
  warnings: string[];
}

export interface Player {
  track_id: number;
  display_name: string;
  team_id: string;
  role: string;
  jersey_number: number | null;
  coverage: Coverage;
  metrics: Record<string, Metric>;
  average_position: { x: number; y: number } | null;
  first_seen_s: number;
  last_seen_s: number;
  event_links: { timestamp_s: number; event_type: string; event_id: string }[];
}

export interface Goalkeeper {
  track_id: number;
  display_name: string;
  team_id: string;
  coverage: Coverage;
  metrics: Record<string, Metric>;
  distribution_map: Record<string, number | boolean | string>[];
  shot_map: Record<string, number | boolean | string>[];
  average_position: { x: number; y: number } | null;
  note: string;
}

export interface Team {
  team_id: string;
  n_players: number;
  coverage: Coverage;
  metrics: Record<string, Metric>;
  average_positions: Record<string, { x: number; y: number }>;
  attack_direction: string;
  attack_direction_confidence: number;
}

export interface TimelineEvent {
  event_id: string;
  type: string;
  timestamp_s: number;
  frame_idx: number;
  team_id: string;
  track_id: number | null;
  player_name: string;
  related_track_id: number | null;
  related_player_name: string;
  confidence: number;
  confidence_band: "high" | "probable" | "uncertain";
  ball_coverage: number;
  ball_state: string;
  half: number;
  clip: {
    frame_start: number;
    frame_end: number;
    time_start_s: number;
    time_end_s: number;
  } | null;
}

export interface Timeline {
  duration_s: number;
  fps: number;
  events: TimelineEvent[];
  possession: {
    start_s: number;
    end_s: number;
    state: string;
    team_id: string;
    track_id: number | null;
    confidence: number;
  }[];
  filters: {
    types: string[];
    teams: string[];
    players: number[];
    confidence_bands: string[];
    halves: number[];
  };
}

export interface Heatmap {
  kind: string;
  track_id: number | null;
  team_id: string | null;
  grid: number[][];
  grid_x: number;
  grid_y: number;
  n_samples: number;
  coverage: number;
  time_range_s: [number, number];
  phase: string;
  half: number | null;
  reportable: boolean;
}

export interface PassingNetwork {
  team_id: string;
  window: string;
  nodes: {
    track_id: number;
    display_name: string;
    x: number | null;
    y: number | null;
    passes: number;
    centrality: number;
    coverage: Coverage;
  }[];
  edges: {
    source: number;
    target: number;
    count: number;
    progressive: number;
    weight: number;
  }[];
  most_influential: number | null;
  most_used_connection: number[] | null;
  isolated_players: number[];
  dominant_side: string;
  n_passes: number;
}

export interface MatchSummary {
  video_id: string;
  context: Record<string, number | string>;
  possession: {
    total_s: number;
    controlled_s: number;
    unknown_s: number;
    determinable_ratio: number;
    unknown_ratio: number;
    states: Record<string, number>;
    teams: Record<
      string,
      { seconds: number; share_of_controlled: number; share_of_match: number }
    >;
  };
  event_counts: Record<string, number>;
  n_players: number;
  n_goalkeepers: number;
  teams: string[];
  data_quality: DataQuality;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail.slice(0, 300)}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/api/health"),

  listProjects: () => request<Project[]>("/api/projects"),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  createProject: (name: string, description = "") =>
    request<Project>("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description }),
    }),
  deleteProject: (id: string) =>
    request<{ deleted: string }>(`/api/projects/${id}`, { method: "DELETE" }),

  uploadMatch: async (
    projectId: string,
    file: File,
    mode: string,
    name: string,
    onProgress?: (fraction: number) => void,
  ): Promise<Job> =>
    new Promise((resolve, reject) => {
      const form = new FormData();
      form.append("file", file);
      form.append("mode", mode);
      form.append("name", name);
      // XHR rather than fetch: upload progress is the one thing fetch still
      // cannot report, and a match file is large enough that a silent
      // multi-minute upload feels broken.
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/projects/${projectId}/jobs`);
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      };
      xhr.onload = () =>
        xhr.status >= 200 && xhr.status < 300
          ? resolve(JSON.parse(xhr.responseText))
          : reject(new Error(`${xhr.status}: ${xhr.responseText.slice(0, 300)}`));
      xhr.onerror = () => reject(new Error("upload failed"));
      xhr.send(form);
    }),

  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobProgress: (id: string) =>
    request<{ id: string; status: string; stage: string; progress: number; error: string }>(
      `/api/jobs/${id}/progress`,
    ),
  deleteJob: (id: string) =>
    request<{ deleted: string }>(`/api/jobs/${id}`, { method: "DELETE" }),

  summary: (id: string) => request<MatchSummary>(`/api/jobs/${id}/summary`),
  players: (id: string, teamId?: string) =>
    request<Player[]>(`/api/jobs/${id}/players${teamId ? `?team_id=${teamId}` : ""}`),
  player: (id: string, trackId: number) =>
    request<Player>(`/api/jobs/${id}/players/${trackId}`),
  goalkeepers: (id: string) => request<Goalkeeper[]>(`/api/jobs/${id}/goalkeepers`),
  teams: (id: string) => request<Team[]>(`/api/jobs/${id}/teams`),
  timeline: (id: string) => request<Timeline>(`/api/jobs/${id}/timeline`),
  heatmaps: (id: string, params: { track_id?: number; team_id?: string; kind?: string } = {}) => {
    const query = new URLSearchParams();
    if (params.track_id !== undefined) query.set("track_id", String(params.track_id));
    if (params.team_id) query.set("team_id", params.team_id);
    if (params.kind) query.set("kind", params.kind);
    return request<Heatmap[]>(`/api/jobs/${id}/heatmaps?${query}`);
  },
  networks: (id: string, teamId?: string) =>
    request<PassingNetwork[]>(`/api/jobs/${id}/networks${teamId ? `?team_id=${teamId}` : ""}`),

  videoUrl: (id: string, kind = "annotated") => `${API_BASE}/api/jobs/${id}/video?kind=${kind}`,
  downloadUrl: (id: string, artefact: string) =>
    `${API_BASE}/api/jobs/${id}/download/${artefact}`,
  csvUrl: (id: string, table: string) => `${API_BASE}/api/jobs/${id}/export/${table}.csv`,
};

/** Format a metric for display, or an em dash when it is not reportable. */
export function formatMetric(metric: Metric | undefined, digits = 1): string {
  if (!metric || !metric.reportable || metric.value === null) return "—";
  const value =
    typeof metric.value === "number" && !Number.isInteger(metric.value)
      ? metric.value.toFixed(digits)
      : String(metric.value);
  return metric.unit && metric.unit !== "count" ? `${value} ${metric.unit}` : value;
}

export function coverageTone(coverage: number): string {
  if (coverage >= 0.7) return "text-emerald-400";
  if (coverage >= 0.4) return "text-amber-400";
  return "text-rose-400";
}

export function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}
