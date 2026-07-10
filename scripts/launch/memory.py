"""Persist and restore the last set of launcher selections.

Stored as JSON next to the launcher (``scripts/launch/.launch_last.json``,
git-ignored). Restoring lets the user re-use their previous configuration as
per-step defaults.
"""

from __future__ import annotations

import dataclasses
import json
import os

from .command import Selection

STORE_PATH = os.path.join(os.path.dirname(__file__), ".launch_last.json")


def save_selection(sel: Selection, path: str = STORE_PATH) -> None:
    """Write the selection to disk. Failures are silently ignored."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(sel), fh, indent=2)
    except OSError:
        pass


def load_selection(path: str = STORE_PATH) -> Selection | None:
    """Load a previously-saved selection, or ``None`` if absent/invalid."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Keep only fields the current dataclass understands, so an older stored
    # file doesn't crash a newer launcher.
    valid = {f.name for f in dataclasses.fields(Selection)}
    filtered = {k: v for k, v in data.items() if k in valid}
    try:
        return Selection(**filtered)
    except TypeError:
        return None


def describe(sel: Selection) -> str:
    """One-line human summary of a stored selection for the 'reuse?' prompt."""
    bits = [f"mode={sel.mode}", f"task={sel.task}"]
    if sel.device_index is not None:
        bits.append(f"gpu={sel.device_index}")
    if sel.viz:
        bits.append(f"viz={','.join(sel.viz)}")
    if sel.mode == "train" and sel.max_iterations is not None:
        bits.append(f"iters={sel.max_iterations}")
    if sel.mode == "play" and sel.checkpoint:
        bits.append(f"ckpt={os.path.basename(sel.checkpoint)}")
    return "  ".join(bits)
