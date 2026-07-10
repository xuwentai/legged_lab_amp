"""Interactive launcher for legged_lab training / play runs.

A small, dependency-light TUI (``rich`` + ``prompt_toolkit``) that guides the
user through picking a mode (train/play), a task, a GPU, a visualizer backend
and the common CLI parameters, then either prints the assembled command or
launches it inside a tmux session.

The launcher deliberately does **not** import isaac/gym/torch, so it starts
instantly. Task discovery is done by statically scanning the source tree.

Run it from the repo root::

    python -m scripts.launch
"""
