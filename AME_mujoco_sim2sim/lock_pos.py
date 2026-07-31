import numpy as np
import mujoco
import mujoco.viewer
import yaml
import os


# =====================================================
# PD Controller
# =====================================================
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


# =====================================================
# LOAD CONFIG
# =====================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "g1_29dof.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

xml_path = cfg["xml_path"]
if not os.path.isabs(xml_path):
    xml_path = os.path.join(script_dir, xml_path)

default_joint_angles = np.array(cfg["default_joint_angles"], dtype=np.float32)
kps = np.array(cfg["stiffness"], dtype=np.float32)
kds = np.array(cfg["damping"], dtype=np.float32)

num_joints = len(default_joint_angles)

print("\nLoaded joints:", num_joints)


# =====================================================
# LOAD MUJOCO
# =====================================================
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("\n==================================================")
print("JOINT ORDER (qpos/qvel order)")
print("==================================================")

mujoco_joint_names = []

for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and name != "floating_base_joint":
        mujoco_joint_names.append(name)

for i, name in enumerate(mujoco_joint_names):
    print(f"[{i:02d}] {name}")

print("\n==================================================")
print("ACTUATOR ORDER (data.ctrl order)")
print("==================================================")

actuator_names = []

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    actuator_names.append(name)
    print(f"[{i:02d}] {name}")

print("\n==================================================")
print("BUILDING JOINT → ACTUATOR MAP")
print("==================================================")

# actuator -> joint mapping (MuJoCo internal)
joint_to_actuator = {}

for act_id in range(model.nu):
    joint_id = model.actuator_trnid[act_id][0]

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )

    joint_to_actuator[joint_name] = act_id
    print(f"{joint_name:35s} --> actuator {act_id}")

print("\n==================================================")
print("RESET SIMULATION")
print("==================================================")

mujoco.mj_resetData(model, data)

# floating base
data.qpos[0:3] = [0.0, 0.0, 0.8]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

# set default pose
data.qpos[7:7+num_joints] = default_joint_angles
data.qvel[:] = 0.0

mujoco.mj_forward(model, data)

print("Initial pose applied.")

# 不限制力矩，避免误判
torque_limit = np.ones(model.nu) * 1000.0


# =====================================================
# VIEWER LOOP
# =====================================================
print("\nStarting viewer. Robot should FALL but joints stay stable.")
print("If joints shake violently → mapping problem confirmed.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # joint state (joint order)
        q = data.qpos[7:7+num_joints]
        dq = data.qvel[6:6+num_joints]

        target = default_joint_angles
        target_dq = np.zeros_like(dq)

        # PD in joint space
        tau_joint = pd_control(target, q, kps, target_dq, dq, kds)

        # =================================================
        # REMAP → actuator space
        # =================================================
        tau_actuator = np.zeros(model.nu)

        for j, joint_name in enumerate(mujoco_joint_names):
            if joint_name not in joint_to_actuator:
                continue

            act_id = joint_to_actuator[joint_name]
            tau_actuator[act_id] = tau_joint[j]

        tau_actuator = np.clip(
            tau_actuator, -torque_limit, torque_limit
        )

        data.ctrl[:] = tau_actuator

        mujoco.mj_step(model, data)
        viewer.sync()