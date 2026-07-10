"""GPU discovery and display via ``nvidia-smi``.

Parses ``nvidia-smi`` into a list of :class:`GpuInfo`, renders a ``rich`` table
of current usage, and recommends the most idle GPU as a sensible default.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

_QUERY = "index,name,memory.used,memory.total,utilization.gpu"


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_total_mb: int
    util_pct: int

    @property
    def mem_used_gb(self) -> float:
        return self.mem_used_mb / 1024.0

    @property
    def mem_total_gb(self) -> float:
        return self.mem_total_mb / 1024.0

    @property
    def mem_frac(self) -> float:
        return self.mem_used_mb / self.mem_total_mb if self.mem_total_mb else 0.0

    @property
    def is_idle(self) -> bool:
        """Heuristic: little memory in use and low utilization."""
        return self.mem_frac < 0.10 and self.util_pct < 10


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query_gpus() -> list[GpuInfo]:
    """Return the list of GPUs, or an empty list if ``nvidia-smi`` is absent."""
    if not nvidia_smi_available():
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    mem_used_mb=int(float(parts[2])),
                    mem_total_mb=int(float(parts[3])),
                    util_pct=int(float(parts[4])),
                )
            )
        except ValueError:
            continue
    return gpus


def recommend(gpus: list[GpuInfo]) -> int | None:
    """Index of the most idle GPU (lowest memory use, then lowest util)."""
    if not gpus:
        return None
    best = min(gpus, key=lambda g: (g.mem_used_mb, g.util_pct))
    return best.index


def _mem_style(g: GpuInfo) -> str:
    if g.mem_frac >= 0.75:
        return "bold red"
    if g.mem_frac >= 0.35:
        return "yellow"
    return "green"


def render_table(gpus: list[GpuInfo], recommended: int | None = None, console: Console | None = None) -> None:
    """Pretty-print current GPU usage. Marks the recommended GPU with ``★``."""
    console = console or Console()
    if not gpus:
        console.print("[yellow]No GPUs detected (nvidia-smi unavailable or returned nothing).[/]")
        return
    if recommended is None:
        recommended = recommend(gpus)

    table = Table(title="GPU usage", header_style="bold cyan")
    table.add_column("", width=2)
    table.add_column("idx", justify="right")
    table.add_column("name")
    table.add_column("mem used / total", justify="right")
    table.add_column("mem%", justify="right")
    table.add_column("util%", justify="right")

    for g in gpus:
        star = "★" if g.index == recommended else ""
        mem_str = f"{g.mem_used_gb:5.1f} / {g.mem_total_gb:4.0f} GB"
        style = _mem_style(g)
        table.add_row(
            star,
            str(g.index),
            g.name,
            f"[{style}]{mem_str}[/]",
            f"[{style}]{g.mem_frac * 100:4.0f}%[/]",
            f"{g.util_pct:3d}%",
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    gpus = query_gpus()
    render_table(gpus)
    print("recommended:", recommend(gpus))
