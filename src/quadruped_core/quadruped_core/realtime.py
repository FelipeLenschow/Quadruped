"""One CPU core for the policy on the robot: real_driver runs there alone at SCHED_FIFO, everything else stays off it."""
import os
import platform

from quadruped_core.config_loader import load_config


def _online_cpus():
    cpus = set()
    with open("/sys/devices/system/cpu/online") as f:
        for part in f.read().strip().split(","):
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def reserved_core():
    """The policy's core, or None off the robot, when disabled, or with a single CPU."""
    cfg = load_config().get("realtime", {}) or {}
    if not cfg.get("enabled", True) or platform.machine().lower() not in ("aarch64", "arm64"):
        return None
    online = _online_cpus()
    core = int(cfg.get("core", -1))
    core = max(online) if core < 0 else core
    return core if len(online) > 1 and core in online else None


def _without(cpus, core):
    rest = set(cpus) - {core}
    return rest or set(cpus)


def avoid_reserved_core():
    """Keep the calling thread, and every process it starts from now on, off the policy's core."""
    core = reserved_core()
    if core is not None:
        os.sched_setaffinity(0, _without(os.sched_getaffinity(0), core))
    return core


def take_reserved_core():
    """Move every other visible thread off the policy's core, then pin this process's threads to it at SCHED_FIFO.

    Threads started afterwards inherit both, so call it before the DDS participants and rclpy start theirs.
    """
    core = reserved_core()
    if core is None:
        return None
    me = str(os.getpid())
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if pid == me:
            continue
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in map(int, tids):
            try:
                cpus = os.sched_getaffinity(tid)
                if core in cpus and len(cpus) > 1:
                    os.sched_setaffinity(tid, _without(cpus, core))
            except OSError:
                pass
    priority = int((load_config().get("realtime", {}) or {}).get("priority", 80))
    for tid in map(int, os.listdir(f"/proc/{me}/task")):
        os.sched_setaffinity(tid, {core})
        try:
            os.sched_setscheduler(tid, os.SCHED_FIFO, os.sched_param(priority))
        except PermissionError:
            print(f"[realtime] No permission for SCHED_FIFO (needs root or rtprio); pinned to core {core} only.")
            break
    return core
