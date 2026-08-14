"""Shared terrain-observation and AME settings for rough AMP locomotion tasks."""

# These are the two rough-terrain experiment switches. Keep their defaults aligned with
# the original AME experiment; set them to False and "z" for an MLP using heights only.
G1_ROUGH_USE_AME_ENCODER = True
G1_ROUGH_HEIGHT_SCAN_MODE = "xyz"

if G1_ROUGH_HEIGHT_SCAN_MODE not in {"xyz", "z"}:
    raise ValueError("G1_ROUGH_HEIGHT_SCAN_MODE must be either 'xyz' or 'z'.")

G1_AME_SCAN_RESOLUTION = 0.04
G1_AME_SCAN_SIZE = (0.4, 0.4)
G1_AME_SCAN_POSITION_OFFSET = (0.2, 0.0, 20.0)
G1_AME_HEIGHT_OFFSET = 0.5
G1_AME_MAP_SCAN_DIM = (11, 11, 3 if G1_ROUGH_HEIGHT_SCAN_MODE == "xyz" else 1)
G1_AME_MAP_SCAN_HISTORY_LENGTH = 1
