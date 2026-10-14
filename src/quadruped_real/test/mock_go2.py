#!/usr/bin/env python3
"""A MuJoCo Go2 on the robot's own low-level DDS topics, to run the real driver against on a PC.

rt/lowstate goes out at 500 Hz (joints, IMU, foot force in raw FSR counts); rt/lowcmd comes in and is applied as
the motors do it, tau = kp (q* - q) + kd (dq* - dq) + tau_ff, after the same CRC check the robot makes (a command
with a bad CRC is ignored). Joints are in the SDK's order: FR, FL, RR, RL, hip/thigh/calf each.

Use the loopback interface on both sides (--interface lo here and on the driver) so nothing reaches a real robot.
"""
import argparse
import os
import threading
import time

import mujoco
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

SDK_JOINTS = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
SDK_FEET = ["FR", "FL", "RR", "RL"]
# poses.lie_flat in config.yaml, per leg: hip (signed left/right, front/rear), thigh, calf.
LIE = {"FL": (0.45, 1.25, -2.6), "FR": (-0.45, 1.25, -2.6), "RL": (0.65, 1.25, -2.6), "RR": (-0.65, 1.25, -2.6)}
FSR_BIAS, FSR_SCALE = 16.0, 2.66  # raw counts with the foot unloaded; newtons per count


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_scene = os.path.join(here, "..", "..", "quadruped_description", "mujoco_menagerie", "unitree_go2", "scene.xml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--scene", default=default_scene)
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run (0: until Ctrl+C)")
    parser.add_argument("--max-torque", type=float, default=45.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in SDK_JOINTS]
    vadr = [model.jnt_dofadr[model.joint(n).id] for n in SDK_JOINTS]
    act = [model.actuator(n.replace("_joint", "")).id for n in SDK_JOINTS]
    touch = [model.sensor(f"{f}_touch").adr[0] for f in SDK_FEET]
    accel = model.sensor("accelerometer").adr[0]

    # Lying on its belly, as the robot is after Sport mode's StandDown.
    data.qpos[0:7] = [0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(SDK_JOINTS):
        data.qpos[qadr[i]] = LIE[name[:2]][i % 3]
    mujoco.mj_forward(model, data)

    ChannelFactoryInitialize(0, args.interface)
    state_pub = ChannelPublisher("rt/lowstate", LowState_)
    state_pub.Init()
    crc = CRC()
    latest = {"cmd": None, "bad_crc": 0, "count": 0}
    lock = threading.Lock()

    def on_cmd(msg: LowCmd_):
        with lock:
            latest["count"] += 1
            if crc.Crc(msg) != msg.crc:
                latest["bad_crc"] += 1
                return
            latest["cmd"] = msg

    cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    cmd_sub.Init(on_cmd, 10)

    state = unitree_go_msg_dds__LowState_()
    dt = model.opt.timestep
    start = time.monotonic()
    step = 0
    last_report = start
    print(f"[mock_go2] MuJoCo Go2 on '{args.interface}', {1 / dt:.0f} Hz, waiting for rt/lowcmd.", flush=True)
    while args.duration <= 0 or data.time < args.duration:
        with lock:
            cmd = latest["cmd"]
        for i in range(12):
            q, dq = data.qpos[qadr[i]], data.qvel[vadr[i]]
            tau = 0.0
            if cmd is not None:
                m = cmd.motor_cmd[i]
                tau = m.kp * (m.q - q) + m.kd * (m.dq - dq) + m.tau
            data.ctrl[act[i]] = np.clip(tau, -args.max_torque, args.max_torque)
        mujoco.mj_step(model, data)
        step += 1

        for i in range(12):
            ms = state.motor_state[i]
            ms.q, ms.dq, ms.tau_est = float(data.qpos[qadr[i]]), float(data.qvel[vadr[i]]), float(data.ctrl[act[i]])
        state.imu_state.quaternion = [float(v) for v in data.qpos[3:7]]
        state.imu_state.gyroscope = [float(v) for v in data.qvel[3:6]]
        state.imu_state.accelerometer = [float(v) for v in data.sensordata[accel:accel + 3]]
        state.foot_force = [int(FSR_BIAS + data.sensordata[a] / FSR_SCALE) for a in touch]
        state.tick = step
        state_pub.Write(state)

        now = time.monotonic()
        if now - last_report >= 2.0:
            last_report = now
            v = data.qvel[0:3]
            with lock:
                print(f"[mock_go2] t={data.time:6.1f}s height {data.qpos[2]:.2f} m  vx {v[0]:+.2f} vy {v[1]:+.2f}  "
                      f"lowcmd {latest['count']} ({latest['bad_crc']} bad CRC)", flush=True)
        ahead = start + data.time - time.monotonic()
        if ahead > 0:
            time.sleep(ahead)


if __name__ == "__main__":
    main()
