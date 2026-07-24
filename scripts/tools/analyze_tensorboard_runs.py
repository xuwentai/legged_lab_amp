#!/usr/bin/env python3
"""Compare scalar metrics from two TensorBoard run directories.

The comparison reports both each run's final window and a fair window ending at
the largest step shared by both runs. It also exports aligned raw series so more
specialized analysis can be performed without loading the event files again.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/success_rate",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Curriculum/terrain_levels",
    "Episode_Termination/base_contact",
    "Episode_Termination/base_height",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/ang_vel_xy_l2",
    "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/feet_slide",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/dof_acc_l2",
    "Episode_Reward/dof_torques_l2",
    "Episode_Reward/foot_stair_intrusion",
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/symmetry",
    "Loss/amp/disc_score",
    "Loss/amp/disc_demo_score",
    "Policy/mean_std",
)


@dataclass(frozen=True)
class Point:
    step: int
    wall_time: float
    value: float


@dataclass
class Run:
    path: Path
    label: str
    commit: str | None
    scalars: dict[str, list[Point]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--label-a", default=None)
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tensorboard_comparison"))
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument("--smooth-window", type=int, default=51)
    parser.add_argument(
        "--step-tag",
        default="Train/mean_reward",
        help="Scalar whose step domain defines the common comparison horizon.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Scalar tags to plot and feature in the report. All tags are always exported to CSV.",
    )
    return parser.parse_args()


def _read_commit(run_dir: Path) -> str | None:
    diff_path = run_dir / "legged_lab_amp.diff"
    if not diff_path.exists():
        return None
    match = re.search(r"--- git commit ---\s+([0-9a-f]{7,40})", diff_path.read_text(errors="replace"))
    return match.group(1) if match else None


def load_run(path: Path, label: str | None) -> Run:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {
        tag: [Point(event.step, event.wall_time, float(event.value)) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }
    return Run(path=path, label=label or path.name, commit=_read_commit(path), scalars=scalars)


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _tail_mean(points: list[Point], end_step: int, fraction: float) -> float:
    eligible = [point for point in points if point.step <= end_step]
    if not eligible:
        return math.nan
    count = max(1, math.ceil(len(eligible) * fraction))
    return _finite_mean([point.value for point in eligible[-count:]])


def _final_tail_mean(points: list[Point], fraction: float) -> float:
    return _tail_mean(points, points[-1].step, fraction) if points else math.nan


def _fmt(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.6g}"


def _smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def write_series_csv(output_path: Path, runs: tuple[Run, Run]) -> None:
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "commit", "tag", "step", "wall_time", "value"])
        for run in runs:
            for tag, points in sorted(run.scalars.items()):
                for point in points:
                    writer.writerow([run.label, run.commit or "", tag, point.step, point.wall_time, point.value])


def build_summary(
    runs: tuple[Run, Run], tail_fraction: float, step_tag: str
) -> tuple[list[dict[str, object]], int]:
    run_a, run_b = runs
    if step_tag not in run_a.scalars or step_tag not in run_b.scalars:
        raise KeyError(f"--step-tag '{step_tag}' must exist in both runs.")
    max_a = run_a.scalars[step_tag][-1].step
    max_b = run_b.scalars[step_tag][-1].step
    common_step = min(max_a, max_b)
    rows: list[dict[str, object]] = []
    for tag in sorted(set(run_a.scalars) | set(run_b.scalars)):
        points_a = run_a.scalars.get(tag, [])
        points_b = run_b.scalars.get(tag, [])
        common_a = _tail_mean(points_a, common_step, tail_fraction)
        common_b = _tail_mean(points_b, common_step, tail_fraction)
        rows.append(
            {
                "tag": tag,
                "count_a": len(points_a),
                "count_b": len(points_b),
                "last_step_a": points_a[-1].step if points_a else None,
                "last_step_b": points_b[-1].step if points_b else None,
                "common_tail_a": common_a,
                "common_tail_b": common_b,
                "common_delta_b_minus_a": common_b - common_a,
                "final_tail_a": _final_tail_mean(points_a, tail_fraction),
                "final_tail_b": _final_tail_mean(points_b, tail_fraction),
            }
        )
    return rows, common_step


def write_summary_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _value_at_or_before(points: list[Point], step: int) -> float:
    candidate = math.nan
    for point in points:
        if point.step > step:
            break
        candidate = point.value
    return candidate


def write_snapshots_csv(output_path: Path, runs: tuple[Run, Run], common_step: int) -> None:
    fractions = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    tags = sorted(set(runs[0].scalars) | set(runs[1].scalars))
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "commit", "progress", "step", "tag", "value"])
        for fraction in fractions:
            step = round(common_step * fraction)
            for run in runs:
                for tag in tags:
                    writer.writerow(
                        [
                            run.label,
                            run.commit or "",
                            fraction,
                            step,
                            tag,
                            _value_at_or_before(run.scalars.get(tag, []), step),
                        ]
                    )


def write_markdown(
    output_path: Path,
    runs: tuple[Run, Run],
    rows: list[dict[str, object]],
    common_step: int,
    featured_tags: list[str],
    tail_fraction: float,
) -> None:
    run_a, run_b = runs
    by_tag = {str(row["tag"]): row for row in rows}
    lines = [
        "# TensorBoard Run Comparison",
        "",
        f"- A: `{run_a.label}` (`{run_a.commit or 'unknown commit'}`)",
        f"- B: `{run_b.label}` (`{run_b.commit or 'unknown commit'}`)",
        f"- Common comparison horizon: step `{common_step}`",
        f"- Tail window: final `{tail_fraction:.0%}` of available samples up to the stated horizon",
        "",
        "| Scalar | A at common horizon | B at common horizon | B - A | A final | B final |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in featured_tags:
        if tag not in by_tag:
            continue
        row = by_tag[tag]
        lines.append(
            f"| `{tag}` | {_fmt(float(row['common_tail_a']))} | {_fmt(float(row['common_tail_b']))} | "
            f"{_fmt(float(row['common_delta_b_minus_a']))} | {_fmt(float(row['final_tail_a']))} | "
            f"{_fmt(float(row['final_tail_b']))} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def write_plots(
    output_path: Path, runs: tuple[Run, Run], tags: list[str], smooth_window: int, common_step: int
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib is unavailable; skipped plot generation.")
        return

    with PdfPages(output_path) as pdf:
        for tag in tags:
            if not any(tag in run.scalars for run in runs):
                continue
            fig, ax = plt.subplots(figsize=(10, 5))
            for run in runs:
                points = run.scalars.get(tag, [])
                if not points:
                    continue
                ax.plot(
                    [point.step for point in points],
                    _smooth([point.value for point in points], smooth_window),
                    label=run.label,
                    linewidth=1.5,
                )
            ax.axvline(common_step, color="black", linestyle="--", linewidth=0.8, label="common horizon")
            ax.set(title=tag, xlabel="training step", ylabel="value")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.tail_fraction <= 1.0:
        raise ValueError("--tail-fraction must be in (0, 1].")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1.")

    runs = (
        load_run(args.run_a, args.label_a),
        load_run(args.run_b, args.label_b),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    featured_tags = list(args.tags) if args.tags else list(DEFAULT_TAGS)

    rows, common_step = build_summary(runs, args.tail_fraction, args.step_tag)
    write_series_csv(output_dir / "series.csv", runs)
    write_summary_csv(output_dir / "summary.csv", rows)
    write_snapshots_csv(output_dir / "snapshots.csv", runs, common_step)
    write_markdown(output_dir / "report.md", runs, rows, common_step, featured_tags, args.tail_fraction)
    write_plots(output_dir / "plots.pdf", runs, featured_tags, args.smooth_window, common_step)

    print(f"Compared {runs[0].label} with {runs[1].label} through common step {common_step}.")
    print(f"Wrote analysis artifacts to {output_dir}")


if __name__ == "__main__":
    main()
