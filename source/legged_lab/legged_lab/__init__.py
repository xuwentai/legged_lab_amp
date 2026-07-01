"""
Python module serving as a project/extension template.
"""

import os

# Register Gym environments.
from .tasks import *

# NOTE: The UI extension (``ui_extension_example.py``) imports ``omni.ext``, which only
# exists while Kit is running. Kit loads it automatically via the
# ``...ui_extension_example`` ``[[python.module]]`` entry in ``config/extension.toml``;
# it is intentionally NOT imported here so that ``import legged_lab`` works in
# headless / pure-Python contexts (e.g. before SimulationApp is created).

LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
