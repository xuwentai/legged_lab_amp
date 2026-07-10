"""Static discovery of tasks, experiment names and checkpoints.

Everything here is filesystem/regex based so the launcher never has to import
``gymnasium`` or spin up Isaac Sim (which is slow). The two sources of truth
are:

* ``source/legged_lab/legged_lab/tasks/**/__init__.py`` — ``gym.register`` ids.
* ``logs/rsl_rl/<experiment_name>/<run>/model_<iter>.pt`` — trained checkpoints.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Repo root = three levels up from this file: scripts/launch/discovery.py -> repo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TASKS_ROOT = os.path.join(REPO_ROOT, "source", "legged_lab", "legged_lab", "tasks")
LOGS_ROOT = os.path.join(REPO_ROOT, "logs", "rsl_rl")

_ID_RE = re.compile(r'id\s*=\s*["\']([^"\']+)["\']')
_RSL_ENTRY_RE = re.compile(r'rsl_rl_cfg_entry_point["\']\s*:\s*f?["\']([^"\']+)["\']')
_EXPERIMENT_RE = re.compile(r'experiment_name\s*=\s*["\']([^"\']+)["\']')
_MODEL_RE = re.compile(r"model_(\d+)\.pt$")


@dataclass
class Task:
    """A single registered gym task."""

    task_id: str
    is_play: bool
    is_debug: bool
    # Absolute path to the __init__.py that registered this task.
    source_file: str
    # Class path string from ``rsl_rl_cfg_entry_point`` (e.g. "pkg.mod:Cls").
    rsl_rl_entry: str | None = None


@dataclass
class TaskSet:
    """Discovered tasks split into train vs. play buckets."""

    train: list[Task] = field(default_factory=list)
    play: list[Task] = field(default_factory=list)

    def train_ids(self) -> list[str]:
        return [t.task_id for t in self.train]

    def play_ids(self) -> list[str]:
        return [t.task_id for t in self.play]


def _iter_init_files(tasks_root: str = TASKS_ROOT):
    for dirpath, _dirnames, filenames in os.walk(tasks_root):
        for name in filenames:
            if name == "__init__.py":
                yield os.path.join(dirpath, name)


def scan_tasks(tasks_root: str = TASKS_ROOT) -> TaskSet:
    """Scan the source tree for ``gym.register`` ids.

    Play tasks are those whose id contains ``-Play-``. Debug variants
    (``-Debug-``) are treated as (non-play) train-side tasks but flagged.
    """
    result = TaskSet()
    seen: set[str] = set()

    for init_file in _iter_init_files(tasks_root):
        try:
            with open(init_file, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        if "gym.register" not in text:
            continue

        # Split on register calls so we can pair each id with its nearby
        # rsl_rl entry point.
        for block in text.split("gym.register")[1:]:
            id_match = _ID_RE.search(block)
            if not id_match:
                continue
            task_id = id_match.group(1)
            if task_id in seen:
                continue
            seen.add(task_id)

            entry_match = _RSL_ENTRY_RE.search(block)
            rsl_rl_entry = entry_match.group(1) if entry_match else None

            is_play = "-Play-" in task_id
            is_debug = "-Debug-" in task_id
            task = Task(
                task_id=task_id,
                is_play=is_play,
                is_debug=is_debug,
                source_file=init_file,
                rsl_rl_entry=rsl_rl_entry,
            )
            if is_play:
                result.play.append(task)
            else:
                result.train.append(task)

    result.train.sort(key=lambda t: t.task_id)
    result.play.sort(key=lambda t: t.task_id)
    return result


_TEMPLATE_RE = re.compile(r"^\{(\w+)?\.?__name__\}\.(.+)$")


def _resolve_entry_file(rsl_rl_entry: str, source_file: str) -> str | None:
    """Resolve an ``rsl_rl_cfg_entry_point`` class path to a .py file.

    Entry points in this repo are f-strings evaluated at import time, so the
    statically-read text still contains the template, e.g.::

        f"{agents.__name__}.rsl_rl_ppo_cfg:G1AmpFlatPPORunnerCfg"
        f"{__name__}.g1_amp_flat_env_cfg:G1AmpFlatEnvCfg"

    ``{__name__}`` is the package of ``source_file`` (the ``__init__.py``'s own
    directory); ``{agents.__name__}`` is its ``agents`` subpackage. We expand
    these against the filesystem so nothing has to be imported. A plain,
    fully-qualified dotted path is also supported as a fallback.
    """
    module_path = rsl_rl_entry.split(":", 1)[0]
    src_dir = os.path.dirname(source_file)

    m = _TEMPLATE_RE.match(module_path)
    if m:
        submodule, rest = m.group(1), m.group(2)
        base_dir = os.path.join(src_dir, submodule) if submodule else src_dir
        candidate = os.path.join(base_dir, *rest.split(".")) + ".py"
        return candidate if os.path.isfile(candidate) else None

    # Fallback: fully-qualified module under source/legged_lab/.
    src_pkg_root = os.path.join(REPO_ROOT, "source", "legged_lab")
    candidate = os.path.join(src_pkg_root, *module_path.split(".")) + ".py"
    return candidate if os.path.isfile(candidate) else None


def task_to_experiment_name(task: Task) -> str | None:
    """Best-effort mapping from a task to its ``experiment_name``.

    Strategy:
    1. Parse the agent cfg file referenced by ``rsl_rl_cfg_entry_point`` and
       grab the last ``experiment_name = "..."`` assignment (handles both the
       class attribute and ``__post_init__`` overrides — the override, defined
       later in the file, wins because we take the last match).
    2. Return ``None`` if it cannot be determined; callers fall back to letting
       the user pick an experiment directory directly.
    """
    if not task.rsl_rl_entry:
        return None
    entry_file = _resolve_entry_file(task.rsl_rl_entry, task.source_file)
    if not entry_file:
        return None
    try:
        with open(entry_file, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None

    # The class named after the ':' suffix is the one that's actually used.
    cls_name = task.rsl_rl_entry.split(":", 1)[1] if ":" in task.rsl_rl_entry else None
    matches = _EXPERIMENT_RE.findall(text)
    if not matches:
        return None

    # If we can locate the specific class, prefer the assignment that appears
    # at/after its definition; otherwise take the last match in the file.
    if cls_name:
        cls_def = re.search(rf"class\s+{re.escape(cls_name)}\b", text)
        if cls_def:
            tail = text[cls_def.start():]
            tail_matches = _EXPERIMENT_RE.findall(tail)
            if tail_matches:
                return tail_matches[0]
    return matches[-1]


# --- checkpoint tree --------------------------------------------------------


@dataclass
class Checkpoint:
    iteration: int
    filename: str
    path: str  # absolute


def list_experiments(logs_root: str = LOGS_ROOT) -> list[str]:
    """List experiment directories under ``logs/rsl_rl`` (newest first)."""
    if not os.path.isdir(logs_root):
        return []
    entries = [
        d for d in os.listdir(logs_root)
        if os.path.isdir(os.path.join(logs_root, d))
    ]
    entries.sort(
        key=lambda d: os.path.getmtime(os.path.join(logs_root, d)),
        reverse=True,
    )
    return entries


def list_runs(experiment: str, logs_root: str = LOGS_ROOT) -> list[str]:
    """List run directories under an experiment (newest first by mtime)."""
    exp_dir = os.path.join(logs_root, experiment)
    if not os.path.isdir(exp_dir):
        return []
    runs = [
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d))
    ]
    runs.sort(key=lambda d: os.path.getmtime(os.path.join(exp_dir, d)), reverse=True)
    return runs


def list_checkpoints(experiment: str, run: str, logs_root: str = LOGS_ROOT) -> list[Checkpoint]:
    """List ``model_<iter>.pt`` files in a run, sorted by iteration ascending.

    Sorting is numeric (natural), so ``model_10000.pt`` sorts *after*
    ``model_2000.pt`` — not lexicographically.
    """
    run_dir = os.path.join(logs_root, experiment, run)
    if not os.path.isdir(run_dir):
        return []
    ckpts: list[Checkpoint] = []
    for name in os.listdir(run_dir):
        m = _MODEL_RE.search(name)
        if not m:
            continue
        ckpts.append(
            Checkpoint(
                iteration=int(m.group(1)),
                filename=name,
                path=os.path.join(run_dir, name),
            )
        )
    ckpts.sort(key=lambda c: c.iteration)
    return ckpts


def latest_checkpoint(experiment: str, run: str, logs_root: str = LOGS_ROOT) -> Checkpoint | None:
    ckpts = list_checkpoints(experiment, run, logs_root)
    return ckpts[-1] if ckpts else None


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    ts = scan_tasks()
    print("TRAIN tasks:")
    for t in ts.train:
        print(f"  {t.task_id:45s} -> exp={task_to_experiment_name(t)}")
    print("PLAY tasks:")
    for t in ts.play:
        print(f"  {t.task_id}")
    print("Experiments:", list_experiments())
