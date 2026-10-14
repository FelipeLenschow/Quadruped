#!/usr/bin/env python3
"""C++ Controller against quadruped_core's LocomotionPipeline, step by step, under random ROS inputs.

Both get the same robot states and the same /pipeline, /pose, /safety, /cmd_vel, /gait_command and trick
messages, on one fake clock. Targets, torque and the whole safety / mode / policy state are compared every step.
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

_real_sleep = time.sleep


class Clock:
    t = 0.0


CLOCK = Clock()
time.time = lambda: CLOCK.t
time.monotonic = lambda: CLOCK.t

from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from std_msgs.msg import Bool, Float32, Float32MultiArray, String  # noqa: E402

import quadruped_control as qc  # noqa: E402
from quadruped_control.policies import entry_for, load_spec, read_catalog  # noqa: E402
from quadruped_core import paths  # noqa: E402
from quadruped_core.config_loader import load_policies  # noqa: E402
import quadruped_core.controller.policy_manager as policy_manager  # noqa: E402
import quadruped_control.pipeline as cpp_pipeline  # noqa: E402
from quadruped_core.pipeline import LocomotionPipeline  # noqa: E402


class FakeNode:
    """Enough of rclpy.Node for the pipeline: callbacks by topic, publishers that keep their last message."""

    def __init__(self):
        self.subs, self.pubs, self.logs = {}, {}, []
        logger = SimpleNamespace(**{lvl: (lambda msg, lvl=lvl: self.logs.append((lvl, msg)))
                                    for lvl in ("debug", "info", "warn", "warning", "error")})
        self.get_logger = lambda: logger
        stamp = SimpleNamespace(to_msg=lambda: TimeMsg())
        self.get_clock = lambda: SimpleNamespace(now=lambda: stamp)

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs.setdefault(topic, []).append(cb)

    def create_publisher(self, msg_type, topic, qos):
        pub = SimpleNamespace(last=None)
        pub.publish = lambda msg, pub=pub: setattr(pub, "last", msg)
        self.pubs[topic] = pub
        return pub

    def send(self, topic, msg):
        for cb in self.subs.get(topic, []):
            cb(msg)


def rot(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


class World:
    """Robot states in segments: standing, walking, tilted, pitched, out of range, lying still, upside down."""

    def __init__(self, rng, stance, lie):
        self.rng, self.stance, self.lie = rng, stance, lie
        self.left = 0.0
        self.kind = "stand"

    def sample(self, dt):
        rng = self.rng
        if self.left <= 0.0:
            self.kind = rng.choice(["stand", "walk", "tilt", "pitch", "rom", "lying", "fallen"],
                                   p=[0.3, 0.3, 0.08, 0.07, 0.07, 0.15, 0.03])
            self.left = rng.uniform(0.3, 3.0)
            self.axis = rng.normal(size=3)
            self.joint = int(rng.integers(12))
            self.sign = rng.choice([-1.0, 1.0])
        self.left -= dt
        k = self.kind
        quat = rot(self.axis, rng.normal(scale=0.03))
        q = self.stance + rng.normal(scale=0.05, size=12)
        dq = rng.normal(scale=0.3, size=12)
        contact = np.ones(4)
        if k == "walk":
            q = self.stance + rng.normal(scale=0.2, size=12)
            dq = rng.normal(scale=2.0, size=12)
            contact = (rng.random(4) > 0.5).astype(float)
        elif k == "tilt":
            quat = rot([1.0, 0.3, 0.0], self.sign * rng.uniform(0.6, 1.0))
        elif k == "pitch":
            quat = rot([0.0, 1.0, 0.0], self.sign * rng.uniform(0.35, 0.9))
        elif k == "rom":
            q[self.joint] = (3.6 if self.joint < 8 else 0.0) if self.sign > 0 else -2.8
        elif k == "lying":
            q = self.lie + rng.normal(scale=0.05, size=12)
            dq = rng.normal(scale=0.1, size=12)
        elif k == "fallen":
            quat = rot([1.0, 0.0, 0.0], np.pi * 0.95)
        gravity_b = -qc.projected_gravity_b(quat)
        return dict(
            q=q, dq=dq, quat=quat, gyro=rng.normal(scale=0.3, size=3),
            accel=gravity_b + rng.normal(scale=0.2, size=3),
            pos=np.array([0.0, 0.0, 0.3]) if rng.random() < 0.5 else None,
            vel=rng.normal(scale=0.3, size=3),
            contact=contact,
        )


class CppSide:
    """What a driver does around the C++ Controller: catalog, policy loading, and the ROS inputs."""

    def __init__(self, cfg, catalog, safety_policy, main_ckpt, specs):
        self.core = qc.Controller(cfg)
        self.catalog, self.specs = catalog, specs
        self.core.set_catalog([(n, entry_for(s, paths.repo(s["path"]))) for n, s in catalog.items()])
        if main_ckpt:
            real = os.path.realpath(main_ckpt)
            alias = next((n for n, s in catalog.items() if s["path"] and os.path.realpath(paths.repo(s["path"])) == real), None)
            spec = catalog[alias] if alias else {"path": main_ckpt, "protections": ("heartbeat", "tilt", "pitch", "rom"),
                                                 "commands": True}
            self.core.add_policy("main", self.spec(main_ckpt), entry_for(spec, main_ckpt))
            self.core.set_main_alias(alias)
        safety = safety_policy if cfg.safety_switch else ""
        if safety and safety in catalog:
            if not self.load(safety):
                safety = ""
        else:
            safety = ""
        self.core.set_safety_policy(safety)
        self.core.set_active_policy("main" if self.core.has_policy("main") else None)

    def spec(self, ckpt):
        if ckpt not in self.specs:
            self.specs[ckpt] = load_spec(ckpt)
        return self.specs[ckpt]

    def load(self, name):
        spec = self.catalog[name]
        path = paths.repo(spec["path"]) if not os.path.isabs(spec["path"]) else spec["path"]
        try:
            self.core.add_policy(name, self.spec(path), entry_for(spec, path))
            return True
        except Exception:
            self.core.set_load_error(name, "could not load (see the driver's log)")
            return False

    def send(self, topic, msg, now):
        c, sp = self.core, self.core.safety
        if topic == "/pipeline/mode":
            c.set_mode(msg.data)
        elif topic == "/pipeline/policy":
            name = msg.data.strip()
            if c.request_policy(name) == qc.PolicyRequest.NEEDS_LOAD:
                self.load(name)
        elif topic == "/pose/command":
            c.pose_command(msg.data.strip().lower())
        elif topic == "/pose/interp_duration":
            c.set_pose_interp_duration(msg.data)
        elif topic == "/cmd_vel":
            c.cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z, now)
        elif topic == "/pipeline/trick":
            c.trick(list(msg.data), now)
        elif topic == "/gait_command":
            c.gait(np.asarray(msg.data, dtype=np.float32))
        elif topic == "/gait_command/override":
            c.gait_override(np.asarray(msg.data, dtype=np.float32), now)
        elif topic == "/safety/heartbeat":
            sp.heartbeat(now)
        elif topic == "/safety/max_torque_percent":
            sp.set_max_torque_percent(msg.data)
        elif topic == "/safety/base_tilt_limit_deg":
            sp.set_base_tilt_limit(msg.data)
        elif topic == "/safety/base_forward_tilt_limit_deg":
            sp.set_forward_tilt_limit(msg.data)
        elif topic == "/safety/joint_rom_safety_margin":
            sp.set_rom_margin(msg.data)
        elif topic == "/safety/watchdog_timeout":
            sp.set_watchdog_timeout(msg.data)
        elif topic == "/safety/reset":
            if msg.data:
                sp.safety_reset()
        elif topic == "/safety/estop":
            if msg.data:
                sp.estop()
        elif topic == "/safety/numb":
            sp.numb(msg.data)
        else:
            raise KeyError(topic)


class WrapperSide:
    """The C++ side behind CppPipeline, driven through its own ROS subscriptions."""

    def __init__(self, **kwargs):
        self.node = FakeNode()
        self.pipe = cpp_pipeline.CppPipeline(self.node, **kwargs)
        self.core = self.pipe.core

    def send(self, topic, msg, now):
        self.node.send(topic, msg)


def published(node):
    out = {}
    for topic in ("/robot_state", "/pose/status", "/safety/estop_state"):
        pub = node.pubs.get(topic)
        out[topic] = None if pub is None or pub.last is None else pub.last.data
    joint = node.pubs["/commands/joint_commands"].last
    out["/commands/joint_commands"] = None if joint is None else (list(joint.position), list(joint.effort))
    return out


def floats(values):
    m = Float32MultiArray()
    m.data = [float(v) for v in values]
    return m


EVENT_WEIGHTS = (6, 16, 7, 2, 12, 10, 5, 4, 4, 2, 6, 5, 2, 4, 3, 5, 4, 4)


def random_event(rng, names):
    """(topic, message) or ("enter", None) / ("enter_trip", None) / ("reset", None) / ("gap", seconds)."""
    kind = rng.choice([
        "mode_pose", "mode_policy", "pose", "interp", "policy", "teleop", "trick", "gait", "gait_override",
        "estop", "enter", "enter_trip", "numb", "unnumb", "safety_reset", "gap", "param", "sim_reset",
    ], p=np.array(EVENT_WEIGHTS) / sum(EVENT_WEIGHTS))
    if kind == "mode_pose":
        return "/pipeline/mode", String(data="pose")
    if kind == "mode_policy":
        return "/pipeline/mode", String(data=str(rng.choice(["policy", " Policy ", "bogus"], p=[.8, .1, .1])))
    if kind == "pose":
        return "/pose/command", String(data=str(rng.choice(["stand", "sit", "lie_flat", "pushup", "Stand", "bogus"])))
    if kind == "interp":
        return "/pose/interp_duration", Float32(data=float(rng.choice([0.5, 1.0, 2.0, -1.0])))
    if kind == "policy":
        return "/pipeline/policy", String(data=str(rng.choice(names)))
    if kind == "teleop":
        return "teleop", (rng.uniform(0.2, 3.0), [rng.uniform(-1, 1.5), rng.uniform(-0.5, 0.5), rng.uniform(-1.5, 1.5)])
    if kind == "trick":
        return "trick", (rng.uniform(0.2, 4.0), [float(v) for v in rng.integers(0, 2, size=int(rng.integers(1, 4)))])
    if kind in ("gait", "gait_override"):
        n = int(rng.choice([8, 9, 9, 5]))
        values = [rng.uniform(-0.1, 0.05), rng.uniform(0.5, 4), *rng.choice([0, 0.5, 0.75], size=3),
                  rng.uniform(0.03, 0.2), rng.uniform(-0.3, 0.3), rng.uniform(0.25, 0.42), rng.uniform(0.35, 0.75)][:n]
        topic = "/gait_command" if kind == "gait" else "/gait_command/override"
        return topic, floats(values)
    if kind == "estop":
        return "/safety/estop", Bool(data=bool(rng.random() < 0.9))
    if kind == "numb":
        return "/safety/numb", Bool(data=True)
    if kind == "unnumb":
        return "/safety/numb", Bool(data=False)
    if kind == "safety_reset":
        return "/safety/reset", Bool(data=bool(rng.random() < 0.9))
    if kind == "gap":
        return "gap", rng.uniform(0.2, 1.5)
    if kind == "param":
        topic = str(rng.choice(["/safety/base_tilt_limit_deg", "/safety/base_forward_tilt_limit_deg",
                                "/safety/watchdog_timeout", "/safety/max_torque_percent", "/safety/joint_rom_safety_margin"]))
        value = {"/safety/base_tilt_limit_deg": rng.uniform(20, 45), "/safety/base_forward_tilt_limit_deg": rng.uniform(20, 45),
                 "/safety/watchdog_timeout": rng.choice([0.35, 0.5, -1.0]), "/safety/max_torque_percent": rng.uniform(30, 60),
                 "/safety/joint_rom_safety_margin": rng.uniform(0.05, 0.2)}[topic]
        return topic, Float32(data=float(value))
    return kind, None


def snapshot_py(pipe, node):
    sp, pm = pipe.safety_processor, pipe.policy_manager
    pose = pm.policies.get("pose")
    wtw = pipe._wtw_runners()
    return dict(
        mode=pipe.mode, active=pipe.active_policy, pending=pipe._pending_policy, error=pipe.policy_error,
        estop=sp.is_estopped, numb=sp.is_numb, blocked=sp.is_policy_blocked, trip=sp.trip_reason, numb_reason=sp.numb_reason,
        torque=round(sp.active_max_torque, 9), transition=pipe.mode_transition_active,
        safety_active=pipe._safety_elapsed is not None and not sp.is_numb,
        pose=pose.current_pose_name, pose_done=pose.is_complete, available=pm.available,
        loaded=sorted(n for n in pm.policies if n != "pose"),
        gait=None if not wtw else [round(float(v), 6) for v in wtw[0].gait_command],
        trick=[float(v) for v in pipe.trick_command], cmd=[float(v) for v in pipe.cmd_vel],
    )


def snapshot_cpp(c):
    sp = c.safety
    g = c.gait_command
    return dict(
        mode=c.mode, active=c.active_policy, pending=c.pending_policy, error=c.policy_error,
        estop=sp.is_estopped, numb=sp.is_numb, blocked=sp.is_policy_blocked, trip=sp.trip_reason, numb_reason=sp.numb_reason,
        torque=round(sp.active_max_torque, 9), transition=c.transition_active, safety_active=c.safety_policy_active,
        pose=c.pose.current_pose_name, pose_done=c.pose.is_complete, available=c.available, loaded=sorted(c.loaded),
        gait=None if g is None else [round(float(v), 6) for v in g],
        trick=[float(v) for v in c.trick_command(CLOCK.t)], cmd=[float(v) for v in c.current_cmd_vel(CLOCK.t)],
    )


def run(args, sim_dt, decimation, safety_switch, use_estimator, main_name, seed, specs, safety_policy=None,
        safety_timeout=None, forward_limit=30.0):
    rng = np.random.default_rng(seed)
    CLOCK.t = 1000.0
    policies = load_policies()
    if safety_policy is not None:
        policies = {**policies, "safety_policy": safety_policy}
    raw_catalog = dict(policies.get("policies") or {})
    raw_catalog["Broken"] = {"path": "IsaacLab_Tasks/missing/model_0.pt"}
    policy_manager.load_policies = lambda: {**policies, "policies": raw_catalog}
    cpp_pipeline.load_policies = policy_manager.load_policies
    catalog = read_catalog(raw_catalog)
    main_ckpt = paths.repo(catalog[main_name]["path"]) if main_name else None

    node = FakeNode()
    pipe = LocomotionPipeline(node, checkpoint=main_ckpt, use_estimator=use_estimator, sim_dt=sim_dt,
                              safety_switch=safety_switch)
    pipe.decimation = decimation
    if safety_timeout is not None:
        pipe.safety_timeout = safety_timeout
    pipe.policy_dt = decimation * sim_dt
    cfg = qc.ControllerConfig.from_yaml(paths.config("config.yaml"), paths.description("config", "go2_kinematics.yaml"),
                                        sim_dt, use_estimator, safety_switch)
    cfg.decimation, cfg.policy_dt = decimation, decimation * sim_dt
    if safety_timeout is not None:
        cfg.safety_policy_timeout = safety_timeout
    if args.via_ros:
        cpp = WrapperSide(checkpoint=main_ckpt, use_estimator=use_estimator, sim_dt=sim_dt, safety_switch=safety_switch)
        cpp.pipe.decimation, cpp.pipe.policy_dt = decimation, decimation * sim_dt
    else:
        cpp = CppSide(cfg, catalog, policies.get("safety_policy") or "", main_ckpt, specs)
    c = cpp.core

    stance = qc.default_stance()
    world = World(rng, stance, np.asarray(cfg.lie_pose))
    names = list(catalog) + ["main", "nope", "Broken"]

    def both(topic, msg):
        node.send(topic, msg)
        cpp.send(topic, msg, CLOCK.t)
        while pipe.policy_manager._loading or (args.via_ros and cpp.pipe._loading):
            _real_sleep(0.005)

    for topic, value in (("/safety/max_torque_percent", 55.0), ("/safety/base_tilt_limit_deg", 30.0),
                         ("/safety/base_forward_tilt_limit_deg", forward_limit), ("/safety/joint_rom_safety_margin", 0.15),
                         ("/safety/watchdog_timeout", 0.35)):
        both(topic, Float32(data=value))

    steps = int(args.seconds / sim_dt)
    gap_until = teleop_until = trick_until = -1.0
    teleop = trick = None
    next_heartbeat = CLOCK.t
    worst = 0.0
    counts = {}
    coverage = {}
    for t in range(steps):
        CLOCK.t += sim_dt
        policy_step = t % decimation == 0
        if CLOCK.t >= next_heartbeat:
            next_heartbeat = CLOCK.t + 0.1
            if CLOCK.t >= gap_until:
                both("/safety/heartbeat", Float32(data=1.0))
        if policy_step and CLOCK.t < teleop_until and t % (decimation * 2) == 0:
            m = Twist()
            m.linear.x, m.linear.y, m.angular.z = teleop
            both("/cmd_vel", m)
        if policy_step and CLOCK.t < trick_until and t % (decimation * 3) == 0:
            both("/pipeline/trick", floats(trick))
        if policy_step and rng.random() < args.event_rate * decimation * sim_dt:
            topic, msg = random_event(rng, names)
            counts[topic] = counts.get(topic, 0) + 1
            if topic == "gap":
                gap_until = CLOCK.t + msg
            elif topic == "teleop":
                teleop_until, teleop = CLOCK.t + msg[0], msg[1]
            elif topic == "trick":
                trick_until, trick = CLOCK.t + msg[0], msg[1]
            elif topic == "enter":
                if pipe.safety_processor._estop_latched:
                    pipe.safety_processor._estop_latched = False
                c.safety.release_estop()
            elif topic == "enter_trip":
                if pipe.safety_processor.is_policy_blocked:
                    pipe.safety_processor.release()
                if c.safety.is_policy_blocked:
                    c.safety.release()
            elif topic == "sim_reset":
                pipe.reset()
                c.reset()
            else:
                both(topic, msg)

        s = world.sample(sim_dt)
        target_py = np.array(pipe.step(dict(s), None, CLOCK.t), dtype=np.float64)
        if args.via_ros:
            targets = np.asarray(cpp.pipe.step(dict(s), None, CLOCK.t))
            res = SimpleNamespace(targets=targets, policy_step=policy_step, pose_status=None)
        else:
            res = c.step(s["q"], s["dq"], s["quat"], s["gyro"], s["accel"], s["pos"], s["vel"], s["contact"],
                         sim_time=CLOCK.t, now=CLOCK.t)
            c.drain_events()
        for name in c.loaded:
            runner = pipe.policy_manager.policies.get(name)
            if runner is not None and hasattr(runner, "last_actions"):
                runner.last_actions[:] = c.runner(name).last_actions

        err = float(np.max(np.abs(target_py - res.targets) / np.maximum(1.0, np.abs(target_py))))
        worst = max(worst, err)
        a, b = snapshot_py(pipe, node), snapshot_cpp(c)
        est = float(np.max(np.abs(np.asarray(pipe.telemetry.estimator.velocity) - c.estimator.velocity)))
        status = node.pubs["/pose/status"].last
        if res.pose_status is not None and (status is None or status.data != res.pose_status):
            a["pose_status"], b["pose_status"] = None if status is None else status.data, res.pose_status
        if args.via_ros:
            pa, pb = published(node), published(cpp.node)
            ja, jb = pa.pop("/commands/joint_commands"), pb.pop("/commands/joint_commands")
            if (ja is None) != (jb is None) or (ja and (not np.allclose(ja[0], jb[0], rtol=0, atol=args.tol * 4)
                                                       or ja[1] != jb[1])):
                a["joint_commands"], b["joint_commands"] = ja, jb
            for topic in pa:
                if pa[topic] != pb[topic]:
                    a[topic], b[topic] = pa[topic], pb[topic]
        if a != b or err > args.tol or est > 1e-9:
            print(f"MISMATCH at t={t} ({t * sim_dt:.3f} s, world '{world.kind}'), target error {err:.2e}, estimator {est:.1e}")
            for k in a:
                if a[k] != b[k]:
                    print(f"  {k}: python {a[k]!r}  c++ {b[k]!r}")
            print("  last python logs:", node.logs[-5:])
            return False
        if res.policy_step:
            node.pubs["/pose/status"].last = None
            if args.via_ros:
                cpp.node.pubs["/pose/status"].last = None
        seen = (("safety policy running", b["safety_active"]), ("numb: " + b["numb_reason"].split(" (")[0], b["numb"]),
                ("trip: " + b["trip"].split(" ")[0], b["blocked"]), ("estop", b["estop"]),
                ("policy error", b["error"] is not None), ("policy " + str(b["active"]), b["mode"] == "policy"),
                ("pose " + b["pose"], b["mode"] == "pose"), ("blend", b["transition"]))
        for key, on in seen:
            if on and res.policy_step:
                coverage[key] = coverage.get(key, 0) + 1
    label = (f"dt {sim_dt} x{decimation}, safety_switch {safety_switch}, estimator {use_estimator}, main {main_name}, "
             f"safety policy {policies.get('safety_policy')}, timeout {safety_timeout or cfg.safety_policy_timeout:.0f} s, seed {seed}")
    print(f"{label}: {steps} steps ({args.seconds:.0f} s) match, max target error {worst:.1e}")
    print(f"  events: {dict(sorted((str(k), v) for k, v in counts.items()))}")
    print(f"  policy steps in: {dict(sorted(coverage.items()))}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--event-rate", type=float, default=1.5, help="random events per second")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--via-ros", action="store_true", help="drive CppPipeline through its ROS callbacks")
    args = parser.parse_args()
    specs = {}
    ok = True
    configs = (
        (0.001, 20, True, False, "Final7", {}),
        (0.02, 1, True, True, "WTW", {}),
        (0.005, 4, False, False, "ClockRough", {}),
        (0.02, 1, True, True, None, {}),
        (0.02, 1, True, False, "Reco2", {"forward_limit": 20.0} if args.via_ros else {"safety_timeout": 2.0, "forward_limit": 20.0}),
        (0.02, 1, True, False, "Final7", {"safety_policy": "Broken"}),
    )
    for i, (sim_dt, decimation, safety_switch, use_estimator, main_name, extra) in enumerate(configs):
        ok = run(args, sim_dt, decimation, safety_switch, use_estimator, main_name, args.seed * 100 + i, specs, **extra) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
