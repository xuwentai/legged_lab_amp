"""Assemble ``train.py`` / ``play.py`` command lines from user selections.

Correctness rules baked in here (see the interactive-launch plan):

* Switching GPU emits **both** ``--device cuda:X`` and the hydra override
  ``agent.device=cuda:X`` — never just one.
* Visualization is expressed only through ``--viz`` (CSV of
  ``kit,newton,rerun,viser,none``). ``--headless`` is deprecated in Isaac Lab
  v3 and conflicts with ``--viz``, so this module **never** emits it.
* ``--video`` requires the ``kit`` visualizer; ``viser`` is a kitless live view
  and must not be combined with ``--video``.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

TRAIN_SCRIPT = "scripts/rsl_rl/train.py"
PLAY_SCRIPT = "scripts/rsl_rl/play.py"

VALID_VIZ = {"kit", "newton", "rerun", "viser", "none"}


@dataclass
class Selection:
    """Everything the user picked, for either mode."""

    mode: str  # "train" | "play"
    task: str
    device_index: int | None = None  # None -> don't emit --device / agent.device

    # visualizer backends (subset of VALID_VIZ). Empty list => omit --viz
    # entirely, which means "default headless" in Isaac Lab v3.
    viz: list[str] = field(default_factory=list)

    # common
    num_envs: int | None = None
    seed: int | None = None
    logger: str | None = None  # tensorboard | wandb | neptune
    log_project_name: str | None = None

    # train-only
    max_iterations: int | None = None
    run_name: str | None = None
    resume: bool = False
    load_run: str | None = None
    load_checkpoint: str | None = None  # for resume (checkpoint file glob/name)
    video: bool = False
    video_interval: int | None = None
    video_length: int | None = None

    # play-only
    checkpoint: str | None = None  # absolute path to model_xxx.pt
    real_time: bool = False

    # free-form hydra overrides, e.g. ["agent.algorithm.entropy_coef=0.02"]
    hydra_overrides: list[str] = field(default_factory=list)


class CommandError(ValueError):
    """Raised when a selection cannot produce a valid command."""


def _validate_viz(viz: list[str]) -> list[str]:
    names = [v.strip().lower() for v in viz if v.strip()]
    invalid = [v for v in names if v not in VALID_VIZ]
    if invalid:
        raise CommandError(f"Invalid --viz value(s): {', '.join(invalid)}. Valid: {', '.join(sorted(VALID_VIZ))}.")
    if "none" in names and len(names) > 1:
        raise CommandError("--viz 'none' cannot be combined with other visualizers.")
    # de-dup preserving order
    seen: set[str] = set()
    ordered = [v for v in names if not (v in seen or seen.add(v))]
    return ordered


def _device_tokens(device_index: int | None) -> list[str]:
    if device_index is None:
        return []
    dev = f"cuda:{device_index}"
    # NOTE: --device and agent.device MUST be paired (README requirement).
    return ["--device", dev, f"agent.device={dev}"]


def skips_device(viz: list[str]) -> bool:
    """Whether ``--device`` selection should be dropped for this visualizer.

    The Newton-backed ``viser`` viewer allocates its own GPU buffers on
    ``cuda:0`` regardless of ``--device``; forcing the simulation onto another
    GPU (e.g. ``cuda:3``) then makes the viewer read across devices and crashes
    with a Warp "illegal memory access" (CUDA error 700). Dropping ``--device``
    keeps the sim on the default ``cuda:0`` too, so both live on one GPU.
    """
    return "viser" in viz


def _device_tokens_for(sel: Selection, viz: list[str]) -> list[str]:
    """Device args, honoring the viser exception (see :func:`skips_device`)."""
    if skips_device(viz):
        return []
    return _device_tokens(sel.device_index)


def _common_tail(sel: Selection) -> list[str]:
    """Args shared by both modes that go before hydra overrides."""
    args: list[str] = []
    if sel.num_envs is not None:
        args += ["--num_envs", str(sel.num_envs)]
    if sel.seed is not None:
        args += ["--seed", str(sel.seed)]
    if sel.logger:
        args += ["--logger", sel.logger]
        if sel.logger in {"wandb", "neptune"} and sel.log_project_name:
            args += ["--log_project_name", sel.log_project_name]
    return args


def build_train_cmd(sel: Selection) -> list[str]:
    if sel.mode != "train":
        raise CommandError("build_train_cmd called with non-train selection")
    viz = _validate_viz(sel.viz)
    if sel.video and "kit" not in viz:
        # Recording needs the kit visualizer; add it implicitly.
        viz = viz + ["kit"] if viz != ["none"] else ["kit"]
        viz = _validate_viz(viz)

    argv: list[str] = ["python", TRAIN_SCRIPT, "--task", sel.task]

    if sel.max_iterations is not None:
        argv += ["--max_iterations", str(sel.max_iterations)]
    if sel.run_name:
        argv += ["--run_name", sel.run_name]
    if sel.resume:
        argv += ["--resume"]
        if sel.load_run:
            argv += ["--load_run", sel.load_run]
        if sel.load_checkpoint:
            argv += ["--checkpoint", sel.load_checkpoint]

    if sel.video:
        argv += ["--video"]
        if sel.video_interval is not None:
            argv += ["--video_interval", str(sel.video_interval)]
        if sel.video_length is not None:
            argv += ["--video_length", str(sel.video_length)]

    argv += _common_tail(sel)

    if viz:
        argv += ["--viz", ",".join(viz)]

    argv += _device_tokens_for(sel, viz)
    argv += list(sel.hydra_overrides)
    return argv


def build_play_cmd(sel: Selection) -> list[str]:
    if sel.mode != "play":
        raise CommandError("build_play_cmd called with non-play selection")
    viz = _validate_viz(sel.viz)

    if sel.video and "kit" not in viz:
        viz = viz + ["kit"] if viz != ["none"] else ["kit"]
        viz = _validate_viz(viz)
    if sel.video and "viser" in viz:
        raise CommandError("--video cannot be combined with the viser (kitless) visualizer.")

    argv: list[str] = ["python", PLAY_SCRIPT, "--task", sel.task]

    if sel.checkpoint:
        argv += ["--checkpoint", sel.checkpoint]
    if sel.video:
        argv += ["--video"]
        if sel.video_length is not None:
            argv += ["--video_length", str(sel.video_length)]
    if sel.real_time:
        argv += ["--real-time"]

    argv += _common_tail(sel)

    if viz:
        argv += ["--viz", ",".join(viz)]

    argv += _device_tokens_for(sel, viz)
    argv += list(sel.hydra_overrides)
    return argv


def build_command(sel: Selection) -> list[str]:
    return build_train_cmd(sel) if sel.mode == "train" else build_play_cmd(sel)


def to_shell_string(argv: list[str]) -> str:
    """Render argv as a copy-pasteable single-line shell command."""
    return " ".join(shlex.quote(a) for a in argv)


def uses_viser(sel: Selection) -> bool:
    return "viser" in _validate_viz(sel.viz)
