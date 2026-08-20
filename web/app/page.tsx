"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import type { Job, Project } from "@/lib/api";
import { api } from "@/lib/api";
import { EmptyState, Spinner } from "@/components/ui";

const MODES = [
  { id: "fast_preview", label: "Fast preview", hint: "Every 3rd frame. Do not quote numbers." },
  { id: "balanced", label: "Balanced", hint: "Every frame. The default." },
  { id: "max_accuracy", label: "Maximum accuracy", hint: "Test-time augmentation. Slow." },
];

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [error, setError] = useState<string>("");
  const [name, setName] = useState("");

  const refresh = useCallback(async () => {
    try {
      setProjects(await api.listProjects());
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll while any job is unfinished, so progress advances without a reload.
  useEffect(() => {
    const running = projects?.some((p) =>
      p.jobs.some((j) => ["pending", "running", "analysing"].includes(j.status)),
    );
    if (!running) return;
    const timer = setInterval(() => void refresh(), 3000);
    return () => clearInterval(timer);
  }, [projects, refresh]);

  async function createProject(event: React.FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    try {
      await api.createProject(name.trim());
      setName("");
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1>Projects</h1>
          <p className="mt-1 text-sm text-slate-400">
            Upload a match, pick an analysis mode, and open the finished analytics.
          </p>
        </div>
        <form onSubmit={createProject} className="flex gap-2">
          <input
            className="input w-64"
            placeholder="New project name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button className="btn-primary" type="submit" disabled={!name.trim()}>
            Create
          </button>
        </form>
      </div>

      {error && (
        <div className="rounded-lg border border-rose-800/60 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
          <div className="mt-1 text-xs text-rose-300/70">
            Is the API running? <code>visionpitch serve</code>
          </div>
        </div>
      )}

      {projects === null && <Spinner label="Loading projects" />}
      {projects?.length === 0 && (
        <EmptyState title="No projects yet" hint="Create one above to upload your first match." />
      )}

      <div className="space-y-6">
        {projects?.map((project) => (
          <ProjectCard key={project.id} project={project} onChange={refresh} />
        ))}
      </div>
    </div>
  );
}

function ProjectCard({ project, onChange }: { project: Project; onChange: () => void }) {
  const [mode, setMode] = useState("balanced");
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  async function upload(file: File) {
    setUploading(true);
    setProgress(0);
    try {
      await api.uploadMatch(project.id, file, mode, file.name.replace(/\.[^.]+$/, ""), setProgress);
      await onChange();
    } catch (e) {
      alert(String(e));
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <section className="card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2>{project.name}</h2>
          <div className="mt-1 text-xs text-slate-500">
            {project.n_jobs} match{project.n_jobs === 1 ? "" : "es"} · created{" "}
            {new Date(project.created_at).toLocaleDateString()}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <select
            className="input w-44"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            title={MODES.find((m) => m.id === mode)?.hint}
          >
            {MODES.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
          <button
            className="btn-primary"
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? `Uploading ${(progress * 100).toFixed(0)}%` : "Upload match"}
          </button>
          <button
            className="btn-danger"
            onClick={async () => {
              if (confirm(`Delete "${project.name}" and all its analyses and files?`)) {
                await api.deleteProject(project.id);
                onChange();
              }
            }}
          >
            Delete
          </button>
        </div>
      </div>

      <p className="mt-2 text-xs text-slate-500">{MODES.find((m) => m.id === mode)?.hint}</p>

      {project.jobs.length > 0 && (
        <div className="mt-5 divide-y divide-slate-800 border-t border-slate-800">
          {project.jobs.map((job) => (
            <JobRow key={job.id} job={job} onChange={onChange} />
          ))}
        </div>
      )}
    </section>
  );
}

function JobRow({ job, onChange }: { job: Job; onChange: () => void }) {
  const done = job.status === "completed" && job.has_analytics;
  const failed = job.status === "failed";
  const tone = done
    ? "bg-emerald-500/15 text-emerald-300"
    : failed
      ? "bg-rose-500/15 text-rose-300"
      : "bg-sky-500/15 text-sky-300";

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 py-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">{job.name || job.video_filename}</span>
          <span className={`chip ${tone}`}>{job.status}</span>
          <span className="text-[11px] text-slate-500">{job.mode}</span>
        </div>
        {!done && !failed && (
          <div className="mt-1.5 flex items-center gap-2">
            <div className="h-1 w-40 overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full bg-emerald-500 transition-all"
                style={{ width: `${Math.max(4, job.progress * 100)}%` }}
              />
            </div>
            <span className="text-[11px] text-slate-500">{job.stage || "queued"}</span>
          </div>
        )}
        {failed && (
          <pre className="mt-1 max-w-2xl overflow-x-auto whitespace-pre-wrap text-[11px] text-rose-300/80">
            {job.error.split("\n").slice(0, 3).join("\n")}
          </pre>
        )}
      </div>
      <div className="flex items-center gap-2">
        {done && (
          <Link className="btn-primary" href={`/matches/${job.id}`}>
            Open analysis
          </Link>
        )}
        <button
          className="btn-ghost"
          onClick={async () => {
            if (confirm("Delete this analysis and its files?")) {
              await api.deleteJob(job.id);
              onChange();
            }
          }}
        >
          Delete
        </button>
      </div>
    </div>
  );
}
