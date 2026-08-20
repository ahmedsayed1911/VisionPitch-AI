"""VisionPitch command line interface.

The Phase 1 deliverable is a single command that takes a video and a mode::

    visionpitch analyse match.mp4 --mode balanced

Everything else -- teams, colours, keepers, attack direction -- is discovered.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from visionpitch import __version__
from visionpitch.common.config import AnalysisMode, load_config
from visionpitch.common.logging import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="VisionPitch AI - football broadcast video analysis (Phase 1).",
)
console = Console()
log = get_logger("cli")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"visionpitch {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    pass


# --------------------------------------------------------------------------- #


@app.command()
def analyse(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input video."),
    mode: AnalysisMode = typer.Option(
        AnalysisMode.BALANCED, "--mode", "-m", help="Analysis mode."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Base config YAML (default configs/default.yaml)."
    ),
    output_dir: Path | None = typer.Option(None, "--output", "-o", help="Output root."),
    start: float | None = typer.Option(None, "--start", help="Start time, seconds."),
    end: float | None = typer.Option(None, "--end", help="End time, seconds."),
    max_frames: int | None = typer.Option(None, "--max-frames", help="Cap frames processed."),
    stride: int | None = typer.Option(None, "--stride", help="Process every Nth frame."),
    no_render: bool = typer.Option(False, "--no-render", help="Skip video output."),
    device: str | None = typer.Option(None, "--device", help="cuda | cuda:0 | cpu."),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option(
        None, "--set", help="Override any config key, e.g. --set detection.imgsz=1920."
    ),
) -> None:
    """Run the full Phase 1 pipeline on a video."""
    from visionpitch.pipeline.runner import Phase1Pipeline

    overrides: dict = {}
    ingestion: dict = {}
    if start is not None:
        ingestion["start_time_s"] = start
    if end is not None:
        ingestion["end_time_s"] = end
    if max_frames is not None:
        ingestion["max_frames"] = max_frames
    if stride is not None:
        ingestion["frame_stride"] = stride
    if ingestion:
        overrides["ingestion"] = ingestion
    if output_dir is not None:
        overrides["storage"] = {"output_dir": str(output_dir)}
    if device is not None:
        overrides["runtime"] = {"device": device}

    config = load_config(
        config_path=config_path, mode=mode, overrides=overrides, cli_sets=list(set_ or [])
    )
    configure_logging(log_level)

    console.rule(f"[bold]VisionPitch AI[/bold] - {mode.value}")
    pipeline = Phase1Pipeline(config)
    result = pipeline.run(video, render=not no_render)

    _print_summary(result)


@app.command()
def showcase(
    run_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Completed run directory."
    ),
    video: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Source video the run was produced from."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output mp4 (default <run_dir>/video/showcase.mp4)."
    ),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="Base config YAML (default configs/showcase.yaml)."
    ),
    mode: AnalysisMode = typer.Option(AnalysisMode.BALANCED, "--mode", "-m"),
    start: float | None = typer.Option(None, "--start", help="Start time, seconds."),
    end: float | None = typer.Option(None, "--end", help="End time, seconds."),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option(
        None,
        "--set",
        help="Override any config key, e.g. visualization.showcase.dot_radius_px=5.",
    ),
) -> None:
    """Render the reference-style showcase overlay from a completed run.

    Reads the stored game-state and calibration tables rather than re-running
    detection, so restyling a 9-minute broadcast costs minutes, not half an hour.
    """
    from visionpitch.pipeline.showcase_render import render_showcase

    overrides: dict = {}
    ingestion: dict = {}
    if start is not None:
        ingestion["start_time_s"] = start
    if end is not None:
        ingestion["end_time_s"] = end
    if ingestion:
        overrides["ingestion"] = ingestion

    if config_path is None:
        default = Path("configs/showcase.yaml")
        config_path = default if default.exists() else None

    config = load_config(
        config_path=config_path, mode=mode, overrides=overrides, cli_sets=list(set_ or [])
    )
    configure_logging(log_level)

    console.rule("[bold]VisionPitch AI[/bold] - showcase render")
    result = render_showcase(run_dir, video, config, output_path=output)

    table = Table(title="Showcase render", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in result.to_dict().items():
        table.add_row(key, str(value))
    console.print(table)


def _print_summary(result) -> None:
    quality = result.reports and result.reports.get("game_state", {})

    table = Table(title="Run summary", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    dq = {
        "Frames processed": len(result.frame_indices),
        "Tracks": result.reports.get("tracking", {}).get("tracks_out"),
        "Game-state rows": len(result.rows),
        "Calibrated frames": (
            f"{100 * (result.reports.get('calibration', {}).get('valid_ratio') or 0):.1f}%"
        ),
        "Ball observed": (
            f"{100 * (result.reports.get('ball_tracking', {}).get('observed_ratio') or 0):.1f}%"
        ),
        "Rows with pitch coords": (
            f"{100 * (quality.get('pitch_coordinate_ratio') or 0):.1f}%" if quality else "-"
        ),
        "Total time": f"{result.stage_timings.get('total', 0):.1f}s",
    }
    for key, value in dq.items():
        table.add_row(key, str(value))
    console.print(table)

    setup = result.match_setup
    if setup and setup.attack_directions:
        setup_table = Table(title="Discovered match setup", header_style="bold")
        setup_table.add_column("Team")
        setup_table.add_column("Attacks")
        setup_table.add_column("Confidence", justify="right")
        setup_table.add_column("Players", justify="right")
        for team in ("A", "B"):
            setup_table.add_row(
                f"Team {team}",
                setup.attack_directions.get(team, "unknown"),
                f"{setup.attack_direction_confidence.get(team, 0):.2f}",
                str(setup.active_players.get(team, "-")),
            )
        console.print(setup_table)

    flags = result.reports.get("game_state", {})
    review = (
        result.reports.get("_review")
        or []
    )
    manifest_flags = []
    try:
        summary = json.loads(Path(result.outputs["summary"]).read_text(encoding="utf-8"))
        manifest_flags = summary.get("data_quality", {}).get("requires_manual_review", [])
    except Exception:  # noqa: BLE001
        pass

    for flag in list(review) + list(manifest_flags):
        console.print(f"[yellow]! {flag}[/yellow]")
    _ = flags

    console.print("\n[bold]Outputs[/bold]")
    for name, path in result.outputs.items():
        console.print(f"  {name:18s} {path}")


# --------------------------------------------------------------------------- #


@app.command()
def evaluate(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Run directory."),
    annotations: Path | None = typer.Option(
        None,
        "--annotations",
        "-a",
        exists=True,
        help="Ground-truth JSON. Omit to report reference-free diagnostics only.",
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Report JSON path."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Evaluate a completed run.

    With ``--annotations`` you get detection, tracking and calibration accuracy.
    Without it you get only the reference-free diagnostics, and the report says
    so explicitly rather than implying the system went unmeasured by choice.
    """
    from visionpitch.evaluation.report import evaluate_run

    configure_logging(log_level)
    report = evaluate_run(run_dir, annotations, output)
    console.print_json(json.dumps(report, indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the API that backs the web dashboard."""
    from visionpitch.api.app import run_server

    console.rule("[bold]VisionPitch AI[/bold] API")
    console.print(f"  docs:      http://{host}:{port}/docs")
    console.print("  dashboard: run [bold]npm run dev[/bold] in web/, then "
                  "http://localhost:3000\n")
    run_server(host=host, port=port, reload=reload)


@app.command("analytics")
def analytics(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Run directory."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run Phase 2 analytics over a completed vision run.

    Reads the stored game state and never touches the video, so this takes about
    a second against a run that took a minute and a half to produce.
    """
    from visionpitch.analytics.runner import run_analytics

    configure_logging(log_level)
    console.rule("[bold]VisionPitch AI[/bold] analytics")
    result = run_analytics(run_dir)

    quality = result.summary.get("data_quality", {})
    table = Table(title="Analytics", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Events", str(result.n_events))
    table.add_row("Possession spans", str(result.n_spans))
    table.add_row("Players", str(result.n_players))
    table.add_row("Goalkeepers", str(result.n_goalkeepers))
    table.add_row("Ball known", f"{quality.get('ball_known_pct', 0)}%")
    table.add_row("Possession determinable", f"{quality.get('possession_determinable_pct', 0)}%")
    table.add_row("Usable player rows", f"{quality.get('valid_player_row_pct', 0)}%")
    table.add_row("Time", f"{result.timings_s.get('total', 0):.2f}s")
    console.print(table)

    for warning in quality.get("warnings", []):
        console.print(f"[yellow]! {warning}[/yellow]")

    console.print("\n[bold]Outputs[/bold]")
    for name, path in result.outputs.items():
        console.print(f"  {name:14s} {path}")


@app.command("analyse-match")
def analyse_match(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    mode: AnalysisMode = typer.Option(AnalysisMode.BALANCED, "--mode", "-m"),
    chunk_frames: int = typer.Option(9000, "--chunk-frames"),
    overlap_frames: int = typer.Option(150, "--overlap-frames"),
    output_dir: Path | None = typer.Option(None, "--output", "-o"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore chunk checkpoints."),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Analyse a full-length match in bounded memory.

    Splits the video into overlapping chunks, processes each independently and
    merges them, re-linking player identities across the seams. Peak memory
    depends on chunk length, not match length. Interrupted runs resume at the
    last completed chunk.
    """
    from visionpitch.pipeline.chunked_runner import ChunkedPipeline

    overrides: dict = {
        "chunking": {
            "enabled": True,
            "chunk_frames": chunk_frames,
            "overlap_frames": overlap_frames,
        }
    }
    if output_dir is not None:
        overrides["storage"] = {"output_dir": str(output_dir)}

    config = load_config(mode=mode, overrides=overrides, cli_sets=list(set_ or []))
    configure_logging(log_level)

    console.rule(f"[bold]VisionPitch AI[/bold] full match — {mode.value}")
    result = ChunkedPipeline(config).run(video, resume=not no_resume)

    table = Table(title="Chunked run", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Chunks", str(len(result.chunks)))
    table.add_row("Identities linked across seams", str(result.merge.identities_linked))
    table.add_row("Unlinked boundary tracks", str(result.merge.unlinked_boundary_tracks))
    table.add_row("Duplicate rows dropped", str(result.merge.duplicate_rows_dropped))
    table.add_row("Tracks", str(result.tracks))
    table.add_row("Game-state rows", str(result.rows))
    table.add_row("Total time", f"{result.timings['total']:.1f}s")
    console.print(table)

    console.print("\n[bold]Outputs[/bold]")
    for name, path in result.outputs.items():
        console.print(f"  {name:14s} {path}")


@app.command()
def benchmark(
    dataset: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Dataset root, e.g. data/eval/player_det."
    ),
    label: str = typer.Option("baseline", "--label", "-l", help="Name for this run."),
    mode: AnalysisMode = typer.Option(AnalysisMode.BALANCED, "--mode", "-m"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    split: str = typer.Option("test", "--split"),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option(None, "--set", help="Config overrides."),
) -> None:
    """Score the detector against an annotated dataset.

    Runs the detection stages directly on labelled images — no video decoding —
    so configurations can be A/B compared in seconds. Use `--set` to vary a
    setting and `--label` to name each variant.
    """
    from visionpitch.evaluation.benchmark import (
        run_detection_benchmark,
        summarise_for_console,
        write_benchmark,
    )
    from visionpitch.evaluation.datasets import YoloDetectionDataset

    configure_logging(log_level)
    config = load_config(mode=mode, cli_sets=list(set_ or []))

    data = YoloDetectionDataset(dataset, split=split)
    info = data.info()
    console.rule(f"[bold]benchmark[/bold] {label} — {info.name}")
    if info.kind == "in_distribution":
        console.print(
            "[yellow]This corpus is the checkpoint's own held-out split. The result "
            "measures in-domain performance and is NOT evidence of generalisation."
            "[/yellow]"
        )

    result = run_detection_benchmark(config, data, label=label)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in summarise_for_console(result):
        table.add_row(key, value)
    console.print(table)

    destination = output or dataset / "benchmarks" / f"{label}.json"
    write_benchmark(result, destination)
    console.print(f"\nwrote {destination}")


@app.command("benchmark-tracking")
def benchmark_tracking(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    label: str = typer.Option("baseline", "--label", "-l"),
    mode: AnalysisMode = typer.Option(AnalysisMode.BALANCED, "--mode", "-m"),
    sequences: int = typer.Option(6, "--sequences", help="How many clips to score."),
    max_frames: int | None = typer.Option(300, "--max-frames", help="Per sequence."),
    output: Path | None = typer.Option(None, "--output", "-o"),
    log_level: str = typer.Option("INFO", "--log-level"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Score tracking against identity-annotated sequences (HOTA, IDF1, MOTA)."""
    from visionpitch.evaluation.benchmark import run_tracking_benchmark, write_benchmark
    from visionpitch.evaluation.datasets import GSRDataset, validate_ground_truth

    configure_logging(log_level)
    config = load_config(mode=mode, cli_sets=list(set_ or []))

    data = GSRDataset(dataset, max_sequences=sequences)
    console.rule(f"[bold]tracking benchmark[/bold] {label} — {data.info().name}")
    console.print(
        f"[dim]{len(data.sequences)} sequence(s), out-of-distribution relative to "
        f"the shipped checkpoints[/dim]"
    )

    for sequence in data.sequences[:3]:
        checks = validate_ground_truth(sequence.ground_truth, require_identity=True)
        if checks["issue_counts"]:
            console.print(f"[yellow]{sequence.name}: {checks['issue_counts']}[/yellow]")

    result = run_tracking_benchmark(
        config, data.sequences, label=label, max_frames_per_sequence=max_frames
    )
    pooled = result.metrics["pooled"]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in ("n_sequences", "HOTA", "DetA", "AssA", "IDF1", "MOTA",
                "id_switches", "fragmentations", "mostly_tracked", "mostly_lost"):
        value = pooled.get(key)
        ci = pooled.get(f"{key}_ci95")
        text = f"{value}" + (f"  [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
        table.add_row(key, text)
    table.add_row("runtime", f"{result.runtime_s:.1f}s")
    console.print(table)

    destination = output or dataset / "benchmarks" / f"tracking_{label}.json"
    write_benchmark(result, destination)
    console.print(f"\nwrote {destination}")


@app.command("benchmark-compare")
def benchmark_compare(
    reports: list[Path] = typer.Argument(..., help="Benchmark JSON files, baseline first."),
) -> None:
    """Compare benchmark runs over the same dataset."""
    from visionpitch.evaluation.benchmark import BenchmarkResult, compare_benchmarks

    loaded = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded.append(BenchmarkResult(**data))

    comparison = compare_benchmarks(loaded)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Run")
    for column in ("mAP50_all", "ball_recall", "ball_precision", "player_recall", "runtime_s"):
        table.add_column(column, justify="right")
    for row in comparison["rows"]:
        table.add_row(
            row["label"],
            *[
                f"{row[c]:.4f}" if isinstance(row.get(c), (int, float)) else "-"
                for c in ("mAP50_all", "ball_recall", "ball_precision", "player_recall",
                          "runtime_s")
            ],
        )
    console.print(table)
    console.print(f"[dim]{comparison['note']}[/dim]")


@app.command()
def inspect(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Print a run's manifest and data-quality summary."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        console.print(f"[red]no manifest.json in {run_dir}[/red]")
        raise typer.Exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    console.print_json(json.dumps(manifest.get("data_quality", {}), indent=2))
    for warning in manifest.get("warnings", []):
        console.print(f"[yellow]! {warning}[/yellow]")


@app.command("correct-teams")
def correct_teams(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    corrections: Path = typer.Option(..., "--corrections", "-c", exists=True),
) -> None:
    """Apply reviewer corrections to a run without re-running detection.

    ``corrections`` is JSON mapping track id to overrides::

        {"12": {"team_id": "B"}, "7": {"role": "goalkeeper", "team_id": "A"}}
    """
    from visionpitch.pipeline.correct import apply_track_corrections

    n = apply_track_corrections(run_dir, corrections)
    console.print(f"applied {n} correction(s) and rebuilt game_state")


if __name__ == "__main__":
    app()
