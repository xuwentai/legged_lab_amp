"""MDP terms for the velocity locomotion environments.

v3 pattern: replace ``from isaaclab.envs.mdp import *`` with explicit imports.
The wildcard forces isaaclab's lazy_export system to resolve ALL submodules
(including binary_joint_actions.py which imports Articulation at module level),
loading pxr C++ bindings before SimulationApp is created and causing a crash.
Explicit imports only trigger the specific submodule that defines each name.
"""

# -- isaaclab standard MDP terms used by this task ---------------------------
from isaaclab.envs.mdp import (
    # observations
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
    projected_gravity,
    # rewards
    action_rate_l2,
    ang_vel_xy_l2,
    flat_orientation_l2,
    illegal_contact,
    is_terminated,
    joint_acc_l2,
    joint_deviation_l1,
    joint_pos_limits,
    joint_torques_l2,
    joint_vel_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
    # terminations
    bad_orientation,
    root_height_below_minimum,
    time_out,
    # events
    apply_external_force_torque,
    push_by_setting_velocity,
    randomize_rigid_body_com,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_scale,
    reset_root_state_uniform,
    # actions cfg
    JointPositionActionCfg,
    # commands cfg
    UniformVelocityCommandCfg,
)

# -- legged_lab-specific MDP terms --------------------------------------------
from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
