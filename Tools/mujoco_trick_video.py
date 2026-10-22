"""Record a MuJoCo clip of a trick policy (Tricks, Handstand) through the deploy pipeline.

eval_mujoco's setup (C++ pipeline, the checkpoint's policies.json entry, supervisor heartbeat, policy mode), with
each switch held on in turn and all off between them. A safety stop is cleared at once, as eval_mujoco does, and
counted per segment. Writes an mp4 (mp4v) and prints the pitch per segment.

Needs rclpy, mujoco, cv2 and quadruped_control, so run it in Docker, with a supervisor, on a domain of its own,
rendering with OSMesa:

    docker exec -i -e ROS_DOMAIN_ID=77 -e MUJOCO_GL=osmesa quadruped_container bash -c '
      unset VIRTUAL_ENV; source /opt/ros/humble/setup.bash
      export PYTHONPATH=/app/src/quadruped_common:/app/src/quadruped_drivers:/app/src/quadruped_operator:$(echo /app/install/quadruped_control/lib/python3*/site-packages):$PYTHONPATH
      cd /app; python3 -m quadruped_operator.supervisor --robot=go2 >/dev/null 2>&1 & SUP=$!
      python3 Tools/mujoco_trick_video.py --checkpoint IsaacLab_Tasks/Handstand/logs/rsl_rl/go2_handstand/Handstand1/model_9999.pt \\
        --labels handstand,hindstand --out IsaacLab_Tasks/Handstand/logs/rsl_rl/go2_handstand/Handstand1/videos/mujoco.mp4
      kill $SUP'
"""

import argparse
import os
import time

import cv2
import mujoco
import numpy as np
import rclpy

from quadruped_control.policies import run_params
from quadruped_drivers.eval_mujoco import MujocoEvaluator


def schedule(switches, labels, on_s, off_s, first_s):
    out = [(first_s, [0.0] * 3, "stand")]
    for i in range(switches):
        command = [0.0] * 3
        command[i] = 1.0
        out += [(on_s, command, labels[i]), (off_s, [0.0] * 3, "stand")]
    return out


class TrickClip(MujocoEvaluator):
    def __init__(self, args, segments, **kwargs):
        self.args, self.segments, self.done = args, segments, False
        super().__init__(**kwargs)

    def _evaluation_loop(self):
        a = self.args
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = a.width, a.height
        renderer = mujoco.Renderer(self.model, a.height, a.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        cam.distance, cam.azimuth, cam.elevation = 1.8, 120.0, -15.0
        writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, a.height))
        self._reset_robot()
        step, every = 0, int(round(1000 / a.fps))
        log = []
        for seconds, command, name in self.segments:
            resets = self.safety_resets
            pitches = []
            for _ in range(int(seconds * 1000)):
                self._clear_safety_stop()
                self.pipeline._call(self.pipeline.core.trick, command, time.monotonic())
                targets = self.pipeline.step(raw_state_kwargs=self._get_raw_sensor_data(), cmd_vel=[0.0] * 4,
                                             sim_time=self.data.time)
                torques = self._pd_torques(targets)
                for i, act in enumerate(self.isaac_ctrl_idx):
                    self.data.ctrl[act] = torques[i]
                mujoco.mj_step(self.model, self.data)
                if step % every == 0:
                    renderer.update_scene(self.data, cam)
                    frame = renderer.render().copy()
                    cv2.putText(frame, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if step % 20 == 0:
                    w, x, y, z = self.data.qpos[3:7]
                    pitches.append(np.degrees(np.arctan2(-2 * (x * z - w * y), 1 - 2 * (x * x + y * y))))
                step += 1
            p = np.array(pitches)
            log.append(f"{name:10s} {seconds:.0f} s: pitch (+ nose down) end {p[-1]:+6.1f} deg, max {p.max():+6.1f},"
                       f" min {p.min():+6.1f}; base z {self.data.qpos[2]:.2f} m;"
                       f" safety stops cleared {self.safety_resets - resets}")
        writer.release()
        print("\n".join(["[TrickClip] " + a.out] + log), flush=True)
        self.done = True


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--labels", default="", help="comma-separated name per switch (default trick, or switch1, ...)")
    p.add_argument("--robot", default="go2")
    p.add_argument("--obs-dim", type=int, default=45)
    p.add_argument("--on", type=float, default=6.0, help="s each switch is held on")
    p.add_argument("--off", type=float, default=4.0, help="s all off after each")
    p.add_argument("--first", type=float, default=2.0, help="s all off at the start")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()

    trick = ((run_params(args.checkpoint, "env.yaml") or {}).get("commands") or {}).get("trick") or {}
    switches = int(trick.get("switches") or 1)
    labels = [s for s in args.labels.split(",") if s] or (["trick"] if switches == 1 else
                                                          [f"switch{i + 1}" for i in range(switches)])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rclpy.init()
    node = TrickClip(args, schedule(switches, labels, args.on, args.off, args.first), robot_type=args.robot,
                     checkpoint=args.checkpoint, obs_dim=args.obs_dim, headless=True, rom_margin=None)
    while not node.done and node.physics_thread.is_alive():
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
