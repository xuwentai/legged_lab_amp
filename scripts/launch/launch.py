"""Interactive orchestration for the legged_lab launcher.

Ties together task discovery, GPU display, command assembly and tmux launching
into a guided, inline prompt flow (``rich`` for display, ``prompt_toolkit`` for
line editing). Run with ``python -m scripts.launch``.
"""

from __future__ import annotations

import os
import sys

# Allow running both as ``python -m scripts.launch`` and
# ``python scripts/launch/launch.py``.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from launch import command as cmd_mod  # type: ignore
    from launch import discovery, gpu, memory, runner  # type: ignore
    from launch.command import Selection  # type: ignore
else:
    from . import command as cmd_mod
    from . import discovery, gpu, memory, runner
    from .command import Selection

from prompt_toolkit import prompt
from prompt_toolkit.application import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# --- generic inline prompt helpers ------------------------------------------


class _Cancel(Exception):
    """Raised to abort the whole flow (Ctrl-C / Ctrl-D / 'q')."""


def _read_line(message: str, default: str = "", completer=None, validator=None) -> str:
    """Read one line of input.

    Uses prompt_toolkit for rich line editing on a real terminal, and falls
    back to the builtin ``input()`` when stdin is not a TTY (piped input, dumb
    terminals, CI) so the launcher still works — and can be driven in tests.
    """
    if not sys.stdin.isatty():
        try:
            raw = input(message)
        except (KeyboardInterrupt, EOFError):
            raise _Cancel()
        raw = raw.strip() or default
        if validator is not None:
            from prompt_toolkit.document import Document

            try:
                validator.validate(Document(raw))
            except ValidationError as exc:
                console.print(f"[red]{exc.message}[/]")
                raise _Cancel()
        return raw
    try:
        return prompt(message, default=default, completer=completer, validator=validator).strip()
    except (KeyboardInterrupt, EOFError):
        raise _Cancel()


def _arrow_select(title: str, options: list[str], default_index: int = 0, descriptions: list[str] | None = None) -> int:
    """Interactive arrow-key selector with paging.

    ↑/↓ (or j/k) move the cursor; ←/→ (or h/l) flip whole pages of 10; digits
    0-9 jump to that position on the current page; Enter confirms. Returns the
    chosen 0-based index. Raises ``_Cancel`` on q/Esc/Ctrl-C. Renders inline
    (does not take over the full screen).
    """
    state = {"cursor": max(0, min(default_index, len(options) - 1))}
    page_size = 10
    n = len(options)
    num_pages = (n + page_size - 1) // page_size

    def _fragments():
        cursor = state["cursor"]
        page = cursor // page_size
        start = page * page_size
        end = min(start + page_size, n)
        header = title if num_pages == 1 else f"{title}  (page {page + 1}/{num_pages})"
        lines = [("class:title", f"{header}\n")]
        for i in range(start, end):
            d = i - start  # 0-9 position within the current page
            selected = i == cursor
            pointer = "❯ " if selected else "  "
            style = "class:selected" if selected else "class:option"
            lines.append((style, f"{pointer}{d}. {options[i]}"))
            if descriptions and descriptions[i]:
                dstyle = "class:desc-selected" if selected else "class:desc"
                lines.append((dstyle, f"   {descriptions[i]}"))
            lines.append(("", "\n"))
        if num_pages > 1:
            hint = "  ↑/↓ move · ←/→ page · Enter select · 0-9 jump · q cancel"
        else:
            hint = "  ↑/↓ move · Enter select · 0-9 jump · q cancel"
        lines.append(("class:hint", hint))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    @kb.add("k")
    def _up(event):
        state["cursor"] = (state["cursor"] - 1) % n

    @kb.add("down")
    @kb.add("c-n")
    @kb.add("j")
    def _down(event):
        state["cursor"] = (state["cursor"] + 1) % n

    def _page(delta: int):
        page = state["cursor"] // page_size
        offset = state["cursor"] % page_size
        new_page = (page + delta) % num_pages
        idx = new_page * page_size + offset
        # clamp to the last item on the target page (last page may be partial)
        page_last = min(new_page * page_size + page_size, n) - 1
        state["cursor"] = min(idx, page_last)

    @kb.add("left")
    @kb.add("h")
    def _prev_page(event):
        _page(-1)

    @kb.add("right")
    @kb.add("l")
    def _next_page(event):
        _page(1)

    @kb.add("home")
    @kb.add("g")
    def _home(event):
        state["cursor"] = 0

    @kb.add("end")
    @kb.add("G")
    def _end(event):
        state["cursor"] = n - 1

    @kb.add("enter")
    def _accept(event):
        event.app.exit(result=state["cursor"])

    @kb.add("q")
    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    def _make_digit_handler(d: int):
        def _handler(event):
            # digit selects the item at position ``d`` on the current page
            idx = (state["cursor"] // page_size) * page_size + d
            if idx < n:
                state["cursor"] = idx
                event.app.exit(result=idx)
        return _handler

    for d in range(page_size):
        kb.add(str(d))(_make_digit_handler(d))

    style = Style.from_dict(
        {
            "title": "bold cyan",
            "selected": "bold reverse",
            "option": "",
            "desc": "italic #888888",
            "desc-selected": "italic #aaaaaa",
            "hint": "#666666",
        }
    )

    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(_fragments), always_hide_cursor=True)])),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    result = app.run()
    if result is None:
        raise _Cancel()
    return result


def select_one(title: str, options: list[str], default_index: int = 0, descriptions: list[str] | None = None) -> int:
    """Choose one option.

    On a real terminal this is an arrow-key menu (↑/↓ + Enter, digits jump). On
    a non-TTY (piped input, tests) or if the interactive UI can't start, it
    falls back to a numbered list read via ``input()``.
    """
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _arrow_select(title, options, default_index=default_index, descriptions=descriptions)
        except _Cancel:
            raise
        except Exception:
            # Fall through to the numbered-list prompt on any UI failure.
            pass

    console.print(f"\n[bold cyan]{title}[/]")
    for i, opt in enumerate(options):
        marker = "[green]›[/]" if i == default_index else " "
        desc = f"  [dim]{descriptions[i]}[/]" if descriptions else ""
        console.print(f"  {marker} [bold]{i + 1}[/]. {opt}{desc}")

    class _V(Validator):
        def validate(self, doc):
            t = doc.text.strip()
            if t == "":
                return
            if not t.isdigit() or not (1 <= int(t) <= len(options)):
                raise ValidationError(message=f"Enter 1-{len(options)} (or blank for default)")

    raw = _read_line(f"Choice [1-{len(options)}] (default {default_index + 1}): ", validator=_V())
    return default_index if raw == "" else int(raw) - 1


def select_many(title: str, options: list[str], default_indices: list[int] | None = None) -> list[int]:
    """Multi-select via comma-separated indices. Blank -> defaults."""
    default_indices = default_indices or []
    console.print(f"\n[bold cyan]{title}[/] [dim](comma-separated, e.g. 1,3)[/]")
    for i, opt in enumerate(options):
        marker = "[green]›[/]" if i in default_indices else " "
        console.print(f"  {marker} [bold]{i + 1}[/]. {opt}")
    default_str = ",".join(str(i + 1) for i in default_indices)

    class _V(Validator):
        def validate(self, doc):
            t = doc.text.strip()
            if t == "":
                return
            for part in t.split(","):
                part = part.strip()
                if not part.isdigit() or not (1 <= int(part) <= len(options)):
                    raise ValidationError(message=f"Each item must be 1-{len(options)}")

    raw = _read_line(f"Choice (default {default_str or 'none'}): ", validator=_V())
    if raw == "":
        return default_indices
    return sorted({int(p.strip()) - 1 for p in raw.split(",") if p.strip()})


def ask_int(message: str, default: int | None) -> int | None:
    class _V(Validator):
        def validate(self, doc):
            t = doc.text.strip()
            if t == "":
                return
            if t.lstrip("-").isdigit() is False:
                raise ValidationError(message="Enter an integer (or blank)")

    d = "" if default is None else str(default)
    raw = _read_line(f"{message} [{d or 'skip'}]: ", validator=_V())
    if raw == "":
        return default
    return int(raw)


def ask_text(message: str, default: str = "") -> str:
    return _read_line(f"{message} [{default}]: ", default="") or default


def ask_bool(message: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    raw = _read_line(f"{message} [{d}]: ").lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


# --- flow steps -------------------------------------------------------------


def _short_task_name(task_id: str) -> str:
    """Compact label for run_name / session, e.g. 'amp_flat_g1'."""
    s = task_id.replace("LeggedLab-Isaac-", "").replace("-Play-v0", "").replace("-v0", "")
    s = s.strip("-").replace("--", "-").replace("-", "_").lower()
    return s or "task"


def choose_mode(prev: Selection | None) -> str:
    default = 0 if not prev or prev.mode == "train" else 1
    idx = select_one("Mode", ["train", "play"], default_index=default,
                     descriptions=["train a policy", "play / visualize a checkpoint"])
    return ["train", "play"][idx]


def choose_task(mode: str, tasks: discovery.TaskSet, prev: Selection | None) -> discovery.Task:
    items = tasks.train if mode == "train" else tasks.play
    if not items:
        console.print("[red]No tasks discovered![/]")
        raise _Cancel()
    ids = [t.task_id for t in items]
    default_index = 0
    if prev and prev.task in ids:
        default_index = ids.index(prev.task)
    idx = select_one(f"Task ({mode})", ids, default_index=default_index)
    return items[idx]


def choose_gpu(prev: Selection | None) -> int | None:
    gpus = gpu.query_gpus()
    if not gpus:
        console.print("[yellow]No GPUs detected — the run will use the default device.[/]")
        return None
    rec = gpu.recommend(gpus)
    gpu.render_table(gpus, recommended=rec, console=console)
    indices = [g.index for g in gpus]
    default_pos = indices.index(rec) if rec in indices else 0
    if prev and prev.device_index in indices:
        default_pos = indices.index(prev.device_index)
    labels = [f"cuda:{g.index}  ({g.mem_used_gb:.1f}/{g.mem_total_gb:.0f}GB, util {g.util_pct}%)" for g in gpus]
    idx = select_one("GPU", labels, default_index=default_pos)
    return gpus[idx].index


def choose_viz(mode: str, prev: Selection | None) -> tuple[list[str], bool]:
    """Return (viz_list, record_video)."""
    if mode == "train":
        # train: headless (default) or record video (kit)
        idx = select_one(
            "Visualization (train)",
            ["headless (no visualizer)", "record video (kit)"],
            default_index=0,
            descriptions=["fastest; recommended for training", "adds --video --viz kit"],
        )
        if idx == 0:
            return [], False
        return ["kit"], True

    # play: richer choices
    options = [
        "record video (kit)",
        "viser live view (browser)",
        "rerun",
        "newton",
        "headless / none",
    ]
    descs = [
        "--video --viz kit; saves to logs/.../videos/play",
        "--viz viser; open printed URL (forward the port over ssh)",
        "--viz rerun",
        "--viz newton",
        "--viz none; export policy only",
    ]
    idx = select_one("Visualization (play)", options, default_index=0, descriptions=descs)
    mapping = [(["kit"], True), (["viser"], False), (["rerun"], False), (["newton"], False), (["none"], False)]
    return mapping[idx]


def choose_logger(prev: Selection | None) -> tuple[str | None, str | None]:
    idx = select_one("Logger", ["tensorboard", "wandb", "neptune", "(none)"], default_index=0)
    logger = [None if i == 3 else v for i, v in enumerate(["tensorboard", "wandb", "neptune", None])][idx]
    project = None
    if logger in ("wandb", "neptune"):
        project = ask_text("  log_project_name", default=(prev.log_project_name if prev else "") or "")
        project = project or None
    return logger, project


def choose_checkpoint(task: discovery.Task) -> str | None:
    """Locate a checkpoint via experiment -> run -> model_*.pt selectors."""
    exp_default = discovery.task_to_experiment_name(task)
    experiments = discovery.list_experiments()
    if not experiments:
        console.print("[yellow]No experiments found under logs/rsl_rl — enter a checkpoint path manually.[/]")
        return ask_text("  checkpoint path", default="") or None

    default_pos = experiments.index(exp_default) if exp_default in experiments else 0
    e_idx = select_one("Experiment", experiments, default_index=default_pos)
    experiment = experiments[e_idx]

    runs = discovery.list_runs(experiment)
    if not runs:
        console.print(f"[yellow]No runs under {experiment}.[/]")
        return ask_text("  checkpoint path", default="") or None
    r_idx = select_one("Run (newest first)", runs, default_index=0)
    run = runs[r_idx]

    ckpts = discovery.list_checkpoints(experiment, run)
    if not ckpts:
        console.print(f"[yellow]No model_*.pt under {experiment}/{run}.[/]")
        return ask_text("  checkpoint path", default="") or None
    # default to the latest (largest iteration)
    labels = [f"{c.filename}  (iter {c.iteration})" for c in ckpts]
    default_pos = len(ckpts) - 1
    c_idx = select_one("Checkpoint (latest highlighted)", labels, default_index=default_pos)
    return ckpts[c_idx].path


def collect_hydra_overrides(prev: Selection | None) -> list[str]:
    default = " ".join(prev.hydra_overrides) if prev and prev.hydra_overrides else ""
    console.print("\n[bold cyan]Hydra overrides[/] [dim](space-separated key=value, blank to skip)[/]")
    console.print("  [dim]e.g. agent.algorithm.entropy_coef=0.02 env.scene.num_envs=2048[/]")
    raw = ask_text("  overrides", default=default)
    return [tok for tok in raw.split() if "=" in tok]


def suggest_run_name(sel: Selection) -> str:
    base = _short_task_name(sel.task)
    parts = [base]
    # include a distinctive hydra override if present
    for ov in sel.hydra_overrides:
        key, _, val = ov.partition("=")
        parts.append(f"{key.split('.')[-1]}_{val}")
        break
    if sel.seed is not None:
        parts.append(f"s{sel.seed}")
    return "_".join(parts)


# --- main -------------------------------------------------------------------


def build_selection() -> Selection:
    prev = memory.load_selection()
    if prev is not None:
        console.print(Panel(f"[dim]{memory.describe(prev)}[/]", title="Last configuration", border_style="dim"))
        if not ask_bool("Reuse last configuration as defaults?", default=True):
            prev = None

    mode = choose_mode(prev)
    task = choose_task(mode, discovery.scan_tasks(), prev)
    device_index = choose_gpu(prev)
    viz, video = choose_viz(mode, prev)

    sel = Selection(mode=mode, task=task.task_id, device_index=device_index, viz=viz, video=video)

    # common
    if mode == "play":
        # sensible defaults per README: 64 for video, 16 for viser
        default_envs = 64 if video else (16 if "viser" in viz else None)
    else:
        default_envs = prev.num_envs if prev else None
    sel.num_envs = ask_int("num_envs", default=default_envs)
    sel.seed = ask_int("seed", default=(prev.seed if prev else None))
    sel.logger, sel.log_project_name = choose_logger(prev)

    if mode == "train":
        sel.max_iterations = ask_int("max_iterations", default=(prev.max_iterations if prev else 50000))
        if video:
            sel.video_interval = ask_int("video_interval", default=(prev.video_interval if prev else 2000))
            sel.video_length = ask_int("video_length", default=(prev.video_length if prev else 200))
        sel.resume = ask_bool("resume from a checkpoint?", default=False)
        if sel.resume:
            ck = choose_checkpoint(task)
            if ck:
                # split into load_run + checkpoint filename for train.py semantics
                sel.load_run = os.path.basename(os.path.dirname(ck))
                sel.load_checkpoint = os.path.basename(ck)
        sel.hydra_overrides = collect_hydra_overrides(prev)
        suggested = suggest_run_name(sel)
        sel.run_name = ask_text("run_name", default=(prev.run_name if prev and prev.run_name else suggested))
    else:  # play
        sel.checkpoint = choose_checkpoint(task)
        sel.real_time = ask_bool("real-time playback?", default=False)
        sel.hydra_overrides = collect_hydra_overrides(prev)

    return sel


def preview_and_execute(sel: Selection) -> None:
    argv = cmd_mod.build_command(sel)

    # summary table
    tbl = Table(title="Summary", show_header=False, border_style="cyan")
    tbl.add_row("mode", sel.mode)
    tbl.add_row("task", sel.task)
    if cmd_mod.skips_device(sel.viz):
        tbl.add_row("device", "[yellow]cuda:0 (viser forces GPU 0 — --device omitted)[/]")
    else:
        tbl.add_row("device", f"cuda:{sel.device_index}" if sel.device_index is not None else "(default)")
    tbl.add_row("viz", ",".join(sel.viz) if sel.viz else "(headless)")
    if sel.mode == "train":
        tbl.add_row("max_iterations", str(sel.max_iterations))
        tbl.add_row("run_name", sel.run_name or "")
    else:
        tbl.add_row("checkpoint", sel.checkpoint or "(auto)")
    if sel.hydra_overrides:
        tbl.add_row("hydra", " ".join(sel.hydra_overrides))
    console.print(tbl)

    note = None
    if cmd_mod.uses_viser(sel):
        note = (
            "[yellow]viser:[/] the Newton viewer allocates on cuda:0, so [cyan]--device[/] is omitted "
            "(sim runs on GPU 0 too — pick GPU 0 as your free card).\n"
            "Open the printed URL; forward the port, e.g. [cyan]ssh -L 8080:localhost:8080 <host>[/]"
        )
    runner.print_command(argv, console=console, note=note)

    memory.save_selection(sel)

    # execute choice
    exec_idx = select_one(
        "Execute",
        ["print command only (copy & run yourself)", "launch in tmux now"],
        default_index=0,
    )
    if exec_idx == 0:
        runner.print_copyable(argv, console=console)
        return

    env_cfg = runner.get_env_config(console=console)
    if env_cfg is None:
        if not ask_bool("Continue with the CURRENT shell environment (no venv activation)?", default=False):
            console.print("[yellow]Aborted tmux launch. The command is printed above to run manually.[/]")
            return
        env_cfg = {"VENV_ACTIVATE": "", "PROXY_URL": ""}
        # if no venv, drop the source step by using ':' as a no-op
        env_cfg["VENV_ACTIVATE"] = ""

    # session name = sanitized run_name (train) or play_<task>_<iter> (play)
    if sel.mode == "train":
        session_base = sel.run_name or _short_task_name(sel.task)
    else:
        it = ""
        if sel.checkpoint:
            import re

            m = re.search(r"model_(\d+)\.pt", sel.checkpoint)
            it = m.group(1) if m else ""
        session_base = f"play_{_short_task_name(sel.task)}_{it}".rstrip("_")

    # handle empty-venv case: build_pane needs a sourceable file; if none, use ':'
    if not env_cfg["VENV_ACTIVATE"]:
        env_cfg = dict(env_cfg)
        env_cfg["VENV_ACTIVATE"] = "/dev/null"  # source /dev/null is a harmless no-op
    runner.run_in_tmux(argv, session_base, env_cfg, console=console)


def main() -> int:
    console.print(Panel.fit("[bold]legged_lab interactive launcher[/]", border_style="cyan"))
    try:
        sel = build_selection()
        preview_and_execute(sel)
    except _Cancel:
        console.print("\n[yellow]Cancelled.[/]")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
