"""Record a play clip of the newest checkpoint of a Backflip run, while that run is still training.

Recording inside train.py makes every iteration ~20x slower on this machine (Kit falls back to CPU
transform updates for all 4096 robots), so this plays each checkpoint on its own with a few robots.
Each clip shows --robots robots in a grid from a fixed camera, all on the same schedule (2 s standing,
5 s doing the trick, in turn), with no pushes unless --pushes. Only the newest checkpoint (of those
every --every iterations) is recorded; ones that went by while a clip was being made are skipped.
--once records the newest checkpoint and exits. Clips land in <run>/videos/progress/model_<iter>.mp4.

    python scripts/rsl_rl/video_watcher.py --log-root logs/rsl_rl/go2_backflip
    python scripts/rsl_rl/video_watcher.py --run-dir logs/rsl_rl/go2_backflip/<run> --once
"""

import argparse
import glob
import os
import re
import shutil
import signal
import subprocess
import sys
import time

MODULE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEP_DT = 0.02

_stop = False
_child = None


def _log(msg: str) -> None:
    print(f"[VideoWatcher] {msg}", flush=True)


def _on_signal(signum, frame):
    global _stop
    _stop = True
    if _child and _child.poll() is None:
        _child.terminate()


def _parent_alive(pid: int) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def target_run(args) -> str | None:
    if args.run_dir:
        return os.path.abspath(args.run_dir)
    runs = [d for d in glob.glob(os.path.join(args.log_root, "*")) if os.path.isdir(d)]
    runs = [d for d in runs if os.path.getctime(d) >= args.after - 5]
    return max(runs, key=os.path.getctime) if runs else None


def checkpoints(run_dir: str, settle: float) -> list[tuple[int, str]]:
    found = []
    for pt in glob.glob(os.path.join(run_dir, "model_*.pt")):
        m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(pt))
        if m and time.time() - os.path.getmtime(pt) >= settle:
            found.append((int(m.group(1)), pt))
    return sorted(found)


def task_of(run_dir: str) -> str:
    """The registered task whose rsl_rl experiment folder holds run_dir (Go2-Backflip -> go2_backflip)."""
    experiment = os.path.basename(os.path.dirname(os.path.abspath(run_dir)))
    for init in glob.glob(os.path.join(MODULE, "source", "*", "*", "tasks", "**", "__init__.py"), recursive=True):
        for task in re.findall(r'id="([\w-]+)"', open(init).read()):
            if task.lower().replace("-", "_") == experiment:
                return task
    return "Go2-Backflip"


def clip_path(run_dir: str, it: int) -> str:
    return os.path.join(run_dir, "videos", "progress", f"model_{it}.mp4")


def record(args, run_dir: str, it: int, pt: str) -> bool:
    global _child
    out = clip_path(run_dir, it)
    tmp = out[:-4] + "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    steps = int(round(args.clip_s / STEP_DT))
    env = os.environ.copy()
    source = os.path.join(MODULE, "source", "backflip")
    env["PYTHONPATH"] = f"{source}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else source
    env.update(
        BACKFLIP_PLAY_EPISODE_S=str(args.clip_s),
        BACKFLIP_PLAY_PUSHES="1" if args.pushes else "0",
        BACKFLIP_PLAY_GRID=str(args.robots),
    )
    cmd = [
        sys.executable, os.path.join(MODULE, "scripts", "rsl_rl", "play.py"),
        f"--task={args.task or task_of(run_dir)}", "--headless", "--video", "--video_overview", f"--video_length={steps}",
        f"--checkpoint={pt}", f"--video_dir={tmp}",
    ]
    log_path = os.path.join(os.path.dirname(out), "video_watcher.log")
    _log(f"recording model_{it} ({steps} steps)")
    start = time.time()
    with open(log_path, "w") as log:
        log.write(f"\n===== model_{it} =====\n")
        log.flush()
        _child = subprocess.Popen(cmd, env=env, cwd=MODULE, stdout=log, stderr=subprocess.STDOUT)
        _child.wait()
    clips = sorted(glob.glob(os.path.join(tmp, "*.mp4")))
    if clips:
        shutil.move(clips[-1], out)
        _log(f"model_{it} -> {os.path.relpath(out, MODULE)} ({time.time() - start:.0f} s)")
    elif not _stop:
        _log(f"model_{it}: no clip came out, see {os.path.relpath(log_path, MODULE)}")
    shutil.rmtree(tmp, ignore_errors=True)
    return bool(clips)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="the training run folder to watch")
    g.add_argument("--log-root", help="experiment folder; watches the newest run started after --after")
    p.add_argument("--after", type=float, default=0.0, help="unix time the run started")
    p.add_argument("--every", type=int, default=500, help="iterations between clips")
    p.add_argument("--clip-s", type=float, default=14.0, help="seconds per clip")
    p.add_argument("--pushes", action="store_true", help="keep the pushes in the clips")
    p.add_argument("--robots", type=int, default=16, help="robots in the clip")
    p.add_argument("--task", default=None, help="default: the task whose log folder holds the run")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--settle", type=float, default=15.0, help="seconds a checkpoint must be untouched")
    p.add_argument("--parent-pid", type=int, default=0, help="exit once this process is gone and nothing is pending")
    p.add_argument("--once", action="store_true", help="record what is due now, then exit")
    args = p.parse_args()
    if args.log_root:
        args.log_root = os.path.abspath(args.log_root)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    _log(f"a clip every {args.every} iterations, {args.robots} robots, {args.clip_s} s, pushes {'on' if args.pushes else 'off'}")

    done = 0
    failed = set()
    while not _stop:
        run_dir = target_run(args)
        ckpts = checkpoints(run_dir, args.settle) if run_dir else []
        training = _parent_alive(args.parent_pid) and not args.once
        due = [(it, pt) for it, pt in ckpts if it > 0 and it % args.every == 0]
        if ckpts and not training:
            due.append(ckpts[-1])
        if due:
            it, pt = max(due)
            if pt not in failed and not os.path.exists(clip_path(run_dir, it)):
                if record(args, run_dir, it, pt):
                    done += 1
                else:
                    failed.add(pt)
                if not args.once:
                    continue
        if not training:
            break
        time.sleep(args.poll)
    _log(f"stopped after {done} clip{'' if done == 1 else 's'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
