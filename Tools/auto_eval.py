"""Evaluate every Nth checkpoint of a training run, while that run is still training.

`Mujoco/eval_mujoco.py` sweeps a checkpoint across commanded velocities and writes
`mujoco_eval_report_<checkpoint>.json` next to it, which the dashboard then charts. Run by hand
it is a post-mortem tool: you train for hours, then evaluate. This watches the checkpoints folder
instead and evaluates each one as it lands, so the sweep for 50k is on screen while 200k is still
training.

Two things make this a separate process rather than a hook inside training:

  * Different interpreters. Training needs the Isaac venv (3.11); eval_mujoco.py imports rclpy,
    which only exists under ROS 2 Humble (3.10). Neither can import the other's modules, so the
    evaluation is spawned through whichever interpreter can `import rclpy` -- see
    `resolve_eval_python`, which also knows how to source /opt/ros/<distro>/setup.bash.
  * Different hardware. The sweep is MuJoCo on CPU with the policy on CPU (see
    Controller/policy_runner.py), so it does not compete with Isaac Sim for the GPU.

Typical use is through `launcher.py` (it answers --log-root/--after/--parent-pid for you), but it
stands alone:

    python3 Tools/auto_eval.py --run-dir IsaacLab_Tasks/Walk/logs/skrl/quadruped_direct/Final6

One evaluation runs at a time, oldest checkpoint first, and a checkpoint that already has a report
is never redone -- so a watcher started late catches up on the run's history, and one restarted
picks up where it left off.
"""

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

try:
    import yaml
except ImportError:                     # only costs the obs_dim auto-detect
    yaml = None

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_SCRIPT = os.path.join(REPO_DIR, "Mujoco", "eval_mujoco.py")
SUPERVISOR = os.path.join(REPO_DIR, "Operator", "supervisor.py")
CKPT_RE = re.compile(r"^agent_(\d+)\.pt$")

_stop = False


def _log(msg):
    print(f"[auto_eval] {msg}", flush=True)


def _on_signal(signum, frame):
    global _stop
    _stop = True
    _log(f"signal {signum} -- stopping after the current evaluation is torn down")


# ---------------------------------------------------------------- interpreter

def eval_env():
    """os.environ as ROS 2 needs to see it. The watcher is normally started from the Isaac venv,
    whose bin directory comes first on PATH and whose python3 has no rclpy -- sourcing a ROS setup
    does not undo that, so the venv is stripped here instead. PYTHONPATH goes too: launcher.py puts
    the Isaac task package on it, which has no business inside the evaluation process."""
    env = os.environ.copy()
    venv = env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    if venv:
        bin_dir = os.path.join(venv, "bin")
        env["PATH"] = os.pathsep.join(
            d for d in env.get("PATH", "").split(os.pathsep) if d.rstrip("/") != bin_dir)
    return env


def _probe(prefix):
    """Can this argv prefix import rclpy? The prefix is everything before the script name, so it
    can be a bare interpreter or a bash wrapper that sources a ROS setup first."""
    try:
        r = subprocess.run(prefix + ["-c", "import rclpy"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=90, env=eval_env())
        return r.returncode == 0
    except Exception:
        return False


def _ros_wrapper(setup_sh):
    # `exec python3 "$@"` with "bash" as $0, so the script and its arguments arrive as $@.
    return ["bash", "-c", f'source "{setup_sh}" >/dev/null 2>&1; exec python3 "$@"', "bash"]


def _candidates(explicit):
    if explicit:
        yield [explicit]
        return
    yield [sys.executable]                                   # already inside ROS (Docker, robot)
    distro = os.environ.get("ROS_DISTRO")
    setups = []
    if distro:
        setups.append(f"/opt/ros/{distro}/setup.bash")
    setups += sorted(glob.glob("/opt/ros/*/setup.bash"), reverse=True)
    for s in setups:
        if os.path.exists(s):
            yield _ros_wrapper(s)
    for venv in ("venv_robot", "venv"):                      # a hand-built ROS venv in the repo
        py = os.path.join(REPO_DIR, venv, "bin", "python")
        if os.path.exists(py):
            yield [py]


def resolve_eval_python(explicit=None):
    """The argv prefix that can run eval_mujoco.py, or None if ROS 2 is not reachable from here.
    Also used by launcher.py to decide whether to offer auto-evaluation at all."""
    seen = []
    for prefix in _candidates(explicit):
        key = " ".join(prefix)
        if key in seen:
            continue
        seen.append(key)
        if _probe(prefix):
            return prefix
    return None


# ---------------------------------------------------------------- run scanning

def _obs_dim(run_dir):
    """Same order launcher.py uses: the policy's input shape, else the env's observation space."""
    if not yaml:
        return 0
    params = os.path.join(run_dir, "params")
    try:
        with open(os.path.join(params, "agent.yaml")) as f:
            d = yaml.load(f, Loader=yaml.UnsafeLoader) or {}
        n = (d.get("models", {}).get("policy", {}).get("input_shape") or [0])[0]
        if n:
            return int(n)
    except Exception:
        pass
    try:
        with open(os.path.join(params, "env.yaml")) as f:
            d = yaml.load(f, Loader=yaml.UnsafeLoader) or {}
        return int(d.get("observation_space") or 0)
    except Exception:
        return 0


_target = None


def target_run(args):
    """The run folder this watcher follows, or None until it exists.

    skrl names the folder after the moment training starts, so with --log-root the folder does not
    exist yet when the watcher is launched: the newest one to appear after --after is it. That
    choice is then latched for the life of the process. launcher.py runs one watcher per
    curriculum segment, and latching is what keeps a finished segment's watcher off the next
    segment's folder -- and off its own folder once the launcher renames it."""
    global _target
    if _target:
        return _target
    if args.run_dir:
        _target = os.path.abspath(args.run_dir)
        return _target
    best = None
    for d in glob.glob(os.path.join(args.log_root, "*")):
        if not os.path.isdir(d):
            continue
        try:
            mtime = os.stat(d).st_mtime
        except OSError:
            continue
        if mtime < args.after - 1:
            continue
        if not best or mtime > best[0]:
            best = (mtime, os.path.abspath(d))
    if best:
        _target = best[1]
        _log(f"watching run {os.path.basename(_target)}")
    return _target


def report_for(ckpt):
    base = os.path.basename(ckpt)[:-3]                       # agent_150000.pt -> agent_150000
    return os.path.join(os.path.dirname(ckpt), f"mujoco_eval_report_{base}.json")


def due_checkpoints(run_dir, interval, settle=0.0):
    """Checkpoints owed an evaluation: the first at `interval` steps, then one every `interval`
    after the last checkpoint that has been evaluated.

    Deliberately not `step % interval == 0`. How often skrl writes a checkpoint is agent.yaml's
    checkpoint_interval, which need not divide the evaluation interval -- a run checkpointing every
    60k steps against `--interval 50000` would then never produce a single report. Reports that
    already exist count as evaluated and set the spacing, so a watcher restarted mid-run continues
    from where the last one stopped instead of re-sweeping the whole history."""
    ckpts = []
    for pt in glob.glob(os.path.join(run_dir, "checkpoints", "agent_*.pt")):
        m = CKPT_RE.match(os.path.basename(pt))
        if m:
            ckpts.append((int(m.group(1)), pt))
    ckpts.sort()

    now = time.time()
    due, last = [], 0
    for step, pt in ckpts:
        if step < last + interval:
            continue
        last = step
        if os.path.exists(report_for(pt)):
            continue
        try:
            if now - os.stat(pt).st_mtime < settle:          # still being written by torch.save
                continue
        except OSError:
            continue
        due.append((step, pt))
    return due


# ---------------------------------------------------------------- evaluation

def _looks_dead(report_path):
    """True when the robot did not move at any commanded speed on any axis. Legitimate for an
    early checkpoint that cannot walk yet, and also exactly what an unarmed pipeline produces --
    worth saying out loud either way, because the report itself looks perfectly healthy."""
    try:
        with open(report_path) as f:
            results = json.load(f).get("results", {})
    except Exception:
        return False
    speeds = [v.get("actual_speed", 0.0) for axis in results.values() if isinstance(axis, dict)
              for v in axis.values() if isinstance(v, dict)]
    return bool(speeds) and all(abs(s) < 0.02 for s in speeds)


def start_supervisor(prefix, robot):
    """Operator/supervisor.py broadcasts the safety heartbeat and the torque limit that
    CommandSafetyProcessor waits for; until the first heartbeat arrives active_max_torque is 0 and
    the PD loop produces no torque at all, so the sweep measures a limp robot. During a deployment
    the Console plays this role -- with nobody at a console, the watcher has to."""
    argv = prefix + [SUPERVISOR, f"--robot={robot}"]
    # Its console output is a 10 Hz status box; in the sweep log that buries everything worth
    # reading. eval_mujoco.py reports whether the heartbeat arrived, which is the part that matters.
    proc = subprocess.Popen(argv, cwd=REPO_DIR, env=eval_env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def evaluate(prefix, ckpt, robot, obs_dim, log_path, extra, supervise=True):
    # -u: the log is meant to be tailed while the sweep runs, and eval_mujoco.py prints its
    # per-speed results with plain print(), which block-buffers into a file.
    argv = prefix + ["-u", EVAL_SCRIPT, f"--robot={robot}", f"--internal_policy={ckpt}",
                     f"--obs_dim={obs_dim}", "--headless"] + extra
    step = os.path.basename(ckpt)
    _log(f"evaluating {step} (obs_dim={obs_dim}, robot={robot}) -> {os.path.basename(log_path)}")
    t0 = time.time()
    proc = sup = None
    try:
        with open(log_path, "a") as log:
            log.write(f"\n{'=' * 70}\n[auto_eval] {time.strftime('%Y-%m-%d %H:%M:%S')}  {ckpt}\n")
            log.flush()
            if supervise:
                sup = start_supervisor(prefix, robot)
            proc = subprocess.Popen(argv, cwd=REPO_DIR, env=eval_env(),
                                    stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if _stop:
                    # The report is only written at the very end of the sweep, so killing here
                    # loses the work but never leaves a half-written report behind.
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    _log(f"{step} cancelled after {time.time() - t0:.0f}s")
                    return False
                time.sleep(1.0)
    except KeyboardInterrupt:
        if proc:
            proc.kill()
        raise
    finally:
        if sup and sup.poll() is None:
            sup.terminate()
            try:
                sup.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sup.kill()
    dt = time.time() - t0
    if proc.returncode == 0 and os.path.exists(report_for(ckpt)):
        _log(f"{step} done in {dt / 60:.1f} min -> {os.path.basename(report_for(ckpt))}")
        if _looks_dead(report_for(ckpt)):
            _log(f"   WARNING: {step} never moved at any commanded speed. Expected of an early "
                 f"checkpoint; if a trained one reads like this, check {log_path} for a safety stop.")
        return True
    _log(f"{step} FAILED after {dt / 60:.1f} min (exit {proc.returncode}); see {log_path}")
    return False


def _parent_alive(pid):
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="the training run folder to watch (holds checkpoints/)")
    g.add_argument("--log-root",
                   help="watch the newest run folder to appear under this root (see --after)")
    p.add_argument("--after", type=float, default=0.0,
                   help="with --log-root, ignore run folders older than this epoch time")
    p.add_argument("--interval", type=int, default=50000,
                   help="evaluate one checkpoint per this many training steps (default 50000)")
    p.add_argument("--robot", default="go2")
    p.add_argument("--obs-dim", type=int, default=0, help="0 = read it from the run's params/")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between scans")
    p.add_argument("--settle", type=float, default=15.0,
                   help="seconds a .pt must be untouched before it counts as fully written")
    p.add_argument("--parent-pid", type=int, default=0,
                   help="exit once this process (the training run) is gone")
    p.add_argument("--eval-python", default=os.environ.get("QUADRUPED_EVAL_PYTHON"),
                   help="interpreter for eval_mujoco.py (default: autodetect a ROS 2 one)")
    p.add_argument("--once", action="store_true",
                   help="evaluate what is pending right now, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be evaluated and exit, without running anything")
    p.add_argument("--use-estimator", action="store_true")
    p.add_argument("--no-supervisor", action="store_true",
                   help="do not run Operator/supervisor.py during each sweep (only when a Console "
                        "or supervisor of your own is already broadcasting the heartbeat)")
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if args.dry_run:
        run_dir = target_run(args)
        if not run_dir:
            _log("no run folder to watch yet")
            return 0
        due = due_checkpoints(run_dir, args.interval)
        _log(f"{run_dir}: {len(due)} checkpoint(s) due at every {args.interval} steps")
        for step, pt in due:
            _log(f"  {os.path.basename(pt)}")
        return 0

    prefix = resolve_eval_python(args.eval_python)
    if not prefix:
        _log("no interpreter here can import rclpy, so eval_mujoco.py cannot run.")
        _log("Start this inside Docker, or pass --eval-python /path/to/a/ros/python.")
        return 2
    _log(f"evaluating with: {' '.join(prefix)}")
    _log(f"every {args.interval} steps; {'run ' + args.run_dir if args.run_dir else 'log root ' + args.log_root}")

    extra = ["--use_estimator"] if args.use_estimator else []
    obs_cache = {}
    done = 0
    announced = set()

    while not _stop:
        work = []
        run_dir = target_run(args)
        if run_dir:
            work = [(run_dir, step, pt)
                    for step, pt in due_checkpoints(run_dir, args.interval, args.settle)]

        if len(work) > 2 and tuple(w[2] for w in work) not in announced:
            announced.add(tuple(w[2] for w in work))
            _log(f"{len(work)} checkpoints due -- working through them oldest first, "
                 f"one at a time")

        if not work:
            if args.once:
                break
            if not _parent_alive(args.parent_pid):
                _log("training process is gone and nothing is pending -- exiting")
                break
            time.sleep(args.poll)
            continue

        run_dir, step, pt = work[0]
        if run_dir not in obs_cache:
            obs_cache[run_dir] = args.obs_dim or _obs_dim(run_dir) or 51
        log_path = os.path.join(run_dir, "checkpoints", "auto_eval.log")
        if evaluate(prefix, pt, args.robot, obs_cache[run_dir], log_path, extra,
                    supervise=not args.no_supervisor):
            done += 1

    _log(f"stopped after {done} evaluation{'' if done == 1 else 's'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
