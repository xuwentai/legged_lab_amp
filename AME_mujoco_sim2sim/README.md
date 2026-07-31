# G1 AMP MuJoCo sim2sim

This directory deploys the current repository task
`LeggedLab-Isaac-AMP-Rough-G1-Play-v0` in MuJoCo.

The maintained entrypoint is `deploy_mujoco_ame.py`. It matches the current
858-dimensional actor observation, including term-major five-frame
proprioceptive history and the 11 x 11 x 3 yaw-aligned terrain map.

## Dependencies

Install the lightweight deployment dependencies in the environment used to run
MuJoCo:

```bash
pip install mujoco numpy pyyaml torch
```

Add `pygame` only when gamepad control is needed:

```bash
pip install pygame
```

## Export the actor

RSL-RL training checkpoints contain the complete training policy container.
The sim2sim entrypoint consumes the TorchScript actor exported by the repository
play script.

Loading a checkpoint with `play.py` automatically writes:

```text
logs/rsl_rl/g1_amp_rough/<run>/exported/policy.pt
```

For example:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-Rough-G1-Play-v0 \
  --num_envs 64 \
  --viz kit \
  --checkpoint logs/rsl_rl/g1_amp_rough/2026-07-23_16-08-24/model_12800.pt
```

## Run sim2sim

The checkpoint path can be passed directly. The script resolves its sibling
`exported/policy.pt`:

```bash
CUDA_VISIBLE_DEVICES=2 
python AME_mujoco_sim2sim/deploy_mujoco_ame.py \
  --policy logs/rsl_rl/g1_amp_rough/2026-07-24_15-28-39/exported/policy.pt
```

Passing the export explicitly is equivalent:

```bash
CUDA_VISIBLE_DEVICES=2 python AME_mujoco_sim2sim/deploy_mujoco_ame.py \
  --policy logs/rsl_rl/g1_amp_rough/2026-07-23_16-08-24/exported/policy.pt
```

Useful options:

```bash
# Fixed velocity command: vx, vy, yaw rate
python AME_mujoco_sim2sim/deploy_mujoco_ame.py \
  --policy <policy-or-checkpoint> --command 0.5 0.0 0.0

# Gamepad command
python AME_mujoco_sim2sim/deploy_mujoco_ame.py \
  --policy logs/rsl_rl/g1_amp_rough/2026-07-24_15-28-39/model_20200.pt --gamepad --dry-run

# Validate XML, observation dimension, JIT loading, and one inference
python AME_mujoco_sim2sim/deploy_mujoco_ame.py \
  --policy <policy-or-checkpoint> --dry-run

# Headless, faster-than-real-time execution
python AME_mujoco_sim2sim/deploy_mujoco_ame.py \
  --policy <policy-or-checkpoint> --headless --no-realtime
```

The other `deploy_*.py` files are legacy experiments from the original AME
repository and do not match this repository's current RSL-RL 5.x policy.
