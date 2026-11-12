"""Sim2sim stairs test for a lidar policy (ClockScan): MuJoCo physics, the C++ policy runner, the lidar fan.

No ROS: each trial puts the robot at the start of a stairs terrain (terrain.py's stairs, narrow or spikes), stands it for
--stand s, then commands --speed forward for --seconds and records whether it got across or fell. The .qpol beside
the checkpoint is exported when missing or older than the .pt.

    python3 -m quadruped_drivers.stairs_mujoco --checkpoint <run>/model_N.pt --terrain narrow --start up
    python3 -m quadruped_drivers.stairs_mujoco --checkpoint <run>/model_N.pt --terrain spikes --tops 0.18,0.14,0.10

On spikes, --tops sweeps the post top size in place of --heights (the post height is --spike_height).

--lidar l1 feeds the policy as on the robot: the L1's own scan pattern (l1_lidar), real_sensors' correction and
lidar_fan's binning (FanInput, moved by the base's velocity: the estimator's with --estimator), in place of the exact
fan.

--video writes every trial to one .mp4 from a camera that follows the robot (MUJOCO_GL=osmesa without a display).
"""

import argparse
import json
import os
import time

import mujoco
import numpy as np
import quadruped_control as qc
from quadruped_common import paths
from quadruped_common.config_loader import load_config
from quadruped_control.export import export_checkpoint
from quadruped_drivers import terrain as terrain_mod
from quadruped_drivers.l1_lidar import L1Fan
from quadruped_drivers.lidar_fan import LidarFan
from quadruped_drivers.motor_model import Go2MotorModel

JOINTS = [f"{leg}_{part}_joint" for part in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
SIM_DT = 0.001
DECIMATION = 20
# Fallen: the trunk touches the terrain, the base turns past FALL_TILT from upright (training resets at 0.8 rad
# from the ground plane, which on 22 cm stairs is itself 0.63 rad steep), or drops under FALL_HEIGHT.
FALL_TILT = 1.3
FALL_HEIGHT = 0.12


def load_spec(checkpoint):
    if checkpoint.endswith(".pt"):
        qpol = os.path.splitext(checkpoint)[0] + ".qpol"
        if not os.path.exists(qpol) or os.path.getmtime(qpol) < os.path.getmtime(checkpoint):
            export_checkpoint(checkpoint, qpol, name=os.path.splitext(os.path.basename(checkpoint))[0])
        checkpoint = qpol
    return qc.PolicySpec.from_file(checkpoint)


def yaw_quat(yaw):
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


class StairsSim:
    def __init__(self, spec, terrain, params, kp, kd, estimator=None, contact_force=13.0, l1_cfg=None):
        self.terrain, self.params = terrain, params
        tmp = []
        scene = terrain_mod.scene_path(paths.description("mujoco_menagerie", "unitree_go2", "scene.xml"), terrain, params, tmp)
        try:
            self.model = mujoco.MjModel.from_xml_path(scene)
        finally:
            for f in tmp:
                os.remove(f)
        self.model.opt.timestep = SIM_DT
        terrain_mod.apply_foot_friction(self.model, params)
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 1.0
            self.model.actuator_biasprm[i, 1] = 0.0
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            self.model.actuator_ctrllimited[i] = 0
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.model.dof_damping[self.model.jnt_dofadr[i]] = 0.0
                self.model.dof_frictionloss[self.model.jnt_dofadr[i]] = 0.01
        self.data = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS])
        self.vadr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINTS])
        self.ctrl = np.array([self.model.actuator(n.replace("_joint", "")).id for n in JOINTS])
        self.motor = Go2MotorModel(JOINTS)
        self.kp, self.kd = kp, kd
        self.spec = spec
        self.runner = qc.PolicyRunner(spec)
        self.fan = LidarFan(self.model, spec.lidar) if spec.lidar else None
        self.l1 = self.fan_in = None
        if self.fan and l1_cfg is not None:
            self.l1 = L1Fan(self.model, self.spec.lidar, *l1_cfg)
        self.qpos0 = np.asarray(spec.desired_qpos, dtype=np.float64)
        self.estimator = estimator
        self.contact_force = contact_force
        self.feet = [self.model.geom(f).id for f in ("FL", "FR", "RL", "RR")]
        self._force = np.zeros(6)
        self._down = np.array([0.0, 0.0, -1.0])
        self._gid = np.zeros(1, dtype=np.int32)

    def ground_below(self, xy, z_top=5.0):
        d = mujoco.mj_ray(self.model, self.data, np.array([xy[0], xy[1], z_top]), self._down,
                          np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, self._gid)
        return z_top - d if d >= 0 else 0.0

    def reset(self, x, y, z_ground, yaw):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (x, y, z_ground + 0.36)
        self.data.qpos[3:7] = yaw_quat(yaw)
        self.data.qpos[self.qadr] = self.qpos0
        mujoco.mj_forward(self.model, self.data)
        self.runner.reset_history()
        if self.estimator is not None:
            self.estimator.reset()
        self.targets = self.qpos0.copy()
        if self.l1:
            self.l1.reset(self.data.time)
            self.fan_in = self.l1.input

    def foot_contact(self):
        force = np.zeros(4)
        for i, c in enumerate(self.data.contact[: self.data.ncon]):
            for k, g in enumerate(self.feet):
                if g in (c.geom1, c.geom2):
                    mujoco.mj_contactForce(self.model, self.data, i, self._force)
                    force[k] += abs(self._force[0])
        return (force > self.contact_force).astype(np.float64)

    def policy_step(self, command):
        d = self.data
        quat = d.qpos[3:7].copy()
        rot = d.xmat[1].reshape(3, 3)
        vel_b = rot.T @ d.qvel[:3]
        gyro = d.qvel[3:6].copy()
        if self.estimator is not None:
            vel_b = self.estimator.update(quat, d.sensor("accelerometer").data.copy(), self.foot_contact(),
                                          d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), gyro)
        scan = None
        if self.l1:
            scan = self.l1.ranges(d, vel_b, JOINTS, d.qpos[self.qadr])
        elif self.fan:
            scan = self.fan.ranges(d)
        actions = self.runner.infer(quat, vel_b, gyro, d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(),
                                    np.asarray(command, dtype=np.float64), DECIMATION * SIM_DT, scan=scan)
        self.targets = actions * self.spec.action_scale + self.qpos0

    def physics_step(self):
        d = self.data
        v = d.qvel[self.vadr]
        tau = self.kp * (self.targets - d.qpos[self.qadr]) - self.kd * v
        d.ctrl[self.ctrl] = self.motor.clip(tau, v)
        mujoco.mj_step(self.model, d)
        if self.l1 and round(d.time / SIM_DT) % 5 == 0:
            self.l1.cast(d)

    def trunk_contact(self):
        for c in self.data.contact[: self.data.ncon]:
            bodies = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            if 1 in bodies and 0 in bodies:
                return True
        return False

    def tilt(self):
        return float(np.arccos(np.clip(self.data.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0)))


def goal(terrain, start, params):
    """Test passed when the base is past x_goal (in the walking direction) and within dz of z_goal."""
    h, d, x0 = params["stair_height"], params["stair_depth"], params["stair_gap"]
    if terrain == "spikes":
        h = params["spike_height"]
        return (terrain_mod.spike_end(params) + 0.5, h, +1) if start == "up" else (x0 - 0.3, h, -1)
    if terrain == "narrow":
        n = int(params["narrow_steps"])
        x1 = x0 + n * d
        return (x1 + 0.5, n * h, +1) if start == "up" else (x0 - 0.3, 0.0, -1)
    n = int(params["stair_steps"])
    x1 = x0 + n * d
    return (x1 + 0.5, n * h, +1) if start == "up" else (x1 + 2.0 + n * d + 0.5, 0.0, +1)


class Video:
    def __init__(self, sim, path, fps=25, size=(480, 640)):
        import cv2

        self.cv2, self.fps = cv2, fps
        self.renderer = mujoco.Renderer(sim.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 1
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.2, 135.0, -20.0
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size[1], size[0]))

    def frame(self, data):
        self.renderer.update_scene(data, self.cam)
        self.writer.write(self.cv2.cvtColor(self.renderer.render(), self.cv2.COLOR_RGB2BGR))

    def close(self):
        self.writer.release()


def run_trial(sim, args, rng, viewer=None, video=None):
    x, y, z, yaw = terrain_mod.stair_spawns(args.terrain, sim.params)[args.start]
    back = rng.uniform(0.0, args.x_noise)
    x -= back * np.cos(yaw)
    y += rng.uniform(-args.y_noise, args.y_noise)
    yaw += rng.uniform(-args.yaw_noise, args.yaw_noise)
    sim.reset(x, y, z, yaw)
    x_goal, z_goal, sign = goal(args.terrain, args.start, sim.params)
    steps = int((args.stand + args.seconds) / SIM_DT)
    result = {"across": False, "fell": False, "time": None}
    for k in range(steps):
        t = k * SIM_DT
        if k % DECIMATION == 0:
            sim.policy_step((args.speed if t >= args.stand else 0.0, 0.0, 0.0))
            base = sim.data.qpos[:3]
            height = base[2] - sim.ground_below(base[:2])
            if sim.tilt() > FALL_TILT or height < FALL_HEIGHT or sim.trunk_contact():
                result.update(fell=True, time=round(t, 2))
                break
            if sign * (base[0] - x_goal) > 0 and abs(base[2] - 0.32 - z_goal) < 0.25:
                result.update(across=True, time=round(t, 2))
                break
            if video is not None and (k // DECIMATION) % 2 == 0:
                video.frame(sim.data)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0.0, DECIMATION * SIM_DT * args.slowdown - 0.002))
        sim.physics_step()
    result["end"] = [round(float(v), 2) for v in sim.data.qpos[:3]]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help=".pt (exported to .qpol beside it) or .qpol")
    p.add_argument("--terrain", choices=("stairs", "narrow", "spikes"), default="narrow")
    p.add_argument("--start", choices=("up", "down"), default="up")
    p.add_argument("--heights", default="0.18", help="comma-separated step heights, m")
    p.add_argument("--tops", default="0.14", help="spikes: comma-separated post top sizes, m")
    p.add_argument("--spike_height", type=float, default=None, help="spikes: post height over the pit, m")
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--stand", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--x_noise", type=float, default=0.3, help="start up to this far further back, m")
    p.add_argument("--y_noise", type=float, default=0.1)
    p.add_argument("--yaw_noise", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lidar", choices=("fan", "l1"), default="fan",
                   help="fan: the policy's exact rays; l1: the L1's pattern through lidar_fan, as on the robot")
    p.add_argument("--estimator", action="store_true", help="base velocity from the C++ state estimator, as on the robot")
    p.add_argument("--viewer", action="store_true", help="watch in MuJoCo's viewer")
    p.add_argument("--slowdown", type=float, default=1.0, help="viewer: wall time per sim time")
    p.add_argument("--out", default=None, help="results .json")
    p.add_argument("--video", default=None, help="record to this .mp4 (one per height: name_<cm>.mp4 if several)")
    args = p.parse_args()

    cfg = load_config()
    spec = load_spec(args.checkpoint)
    if not spec.lidar:
        print(f"[stairs_mujoco] {args.checkpoint} has no lidar input; it runs blind.")
    kp, kd = float(cfg["control"]["kp"]), float(cfg["control"]["kd"])
    rng = np.random.default_rng(args.seed)
    report = {"checkpoint": args.checkpoint, "terrain": args.terrain, "start": args.start, "speed": args.speed,
              "lidar": args.lidar,
              "results": []}
    spikes = args.terrain == "spikes"
    sweep = args.tops if spikes else args.heights
    for h in (float(v) for v in sweep.split(",")):
        params = terrain_mod.resolve(cfg)
        params["spike_top" if spikes else "stair_height"] = h
        if spikes and args.spike_height is not None:
            params["spike_height"] = args.spike_height
        estimator = None
        if args.estimator:
            est_cfg = qc.ControllerConfig.from_yaml(paths.config("config.yaml"),
                                                    paths.description("config", "go2_kinematics.yaml"), SIM_DT, True, False)
            estimator = qc.StateEstimator(qc.Kinematics(est_cfg.geometry), est_cfg.estimator)
        est = cfg.get("state_estimator", {})
        force = float(est.get("contact_threshold", 5.0)) * float(est.get("fsr_scale", 2.66))
        l1_cfg = (cfg["real_lidar"], cfg.get("lidar_fan") or {}) if args.lidar == "l1" else None
        sim = StairsSim(spec, args.terrain, params, kp, kd, estimator, force, l1_cfg)
        trials = []
        if args.viewer:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(sim.model, sim.data) as v:
                for _ in range(args.trials):
                    trials.append(run_trial(sim, args, rng, v))
                    print(trials[-1], flush=True)
        else:
            video = None
            if args.video:
                path = args.video if "," not in sweep else args.video.replace(".mp4", f"_{100 * h:.0f}.mp4")
                video = Video(sim, path)
            for _ in range(args.trials):
                trials.append(run_trial(sim, args, rng, video=video))
            if video:
                video.close()
        across = sum(r["across"] for r in trials) / len(trials)
        fell = sum(r["fell"] for r in trials) / len(trials)
        label = "top" if spikes else "step"
        print(f"{args.terrain} {args.start} {label} {100 * h:.0f} cm: across {across:.2f}, fell {fell:.2f} "
              f"({len(trials)} trials)", flush=True)
        report["results"].append({("top_cm" if spikes else "height_cm"): round(100 * h, 1), "across": across, "fell": fell, "trials": trials})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
