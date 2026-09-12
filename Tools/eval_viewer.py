import os
import json
import math
import glob
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse
from pathlib import Path

import time
import threading

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False

LIVE_WINDOW_S = 180           # a run whose tfevents was touched this recently counts as live
_scalar_cache = {}            # events_path -> (mtime, size, {tag: [[step, wall, value], ...]})
_cache_lock = threading.Lock()


class _LooseLoader(yaml.SafeLoader if yaml else object):
    """params/env.yaml is a pyyaml dump full of !!python/... tags. Keep tuples as lists,
    drop anything else rather than refusing to parse the whole file."""


if yaml:
    def _py_tag(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return None
        return None
    _LooseLoader.add_multi_constructor("tag:yaml.org,2002:python/", _py_tag)
    _LooseLoader.add_multi_constructor("!python/", _py_tag)


def _load_yaml(path):
    if not yaml or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return yaml.load(f, Loader=_LooseLoader) or {}
    except Exception as e:
        print(f"[viewer] could not parse {path}: {e}")
        return {}


def _flatten_cfg(env_cfg, agent_cfg):
    """Pull the fields that actually distinguish one run from another into a flat dict."""
    out = {}
    for k, v in (env_cfg or {}).items():
        if isinstance(v, (int, float, str, bool)) or (
            isinstance(v, list) and len(v) <= 4 and all(isinstance(x, (int, float)) for x in v)
        ):
            out[k] = v
    scene = (env_cfg or {}).get("scene") or {}
    if isinstance(scene, dict) and "num_envs" in scene:
        out["num_envs"] = scene["num_envs"]
    ag = ((agent_cfg or {}).get("agent") or {})
    for k in ("rollouts", "learning_epochs", "mini_batches", "learning_rate",
              "entropy_loss_scale", "value_loss_scale", "rewards_shaper_scale"):
        if k in ag:
            out[f"ppo.{k}"] = ag[k]
    sched = ag.get("learning_rate_scheduler_kwargs") or {}
    if isinstance(sched, dict) and "kl_threshold" in sched:
        out["ppo.kl_threshold"] = sched["kl_threshold"]
    tr = ((agent_cfg or {}).get("trainer") or {})
    if "timesteps" in tr:
        out["ppo.timesteps"] = tr["timesteps"]
    return out


def _find_event_files():
    out = []
    for root in MODULE_DIRS:
        out += glob.glob(os.path.join(root, "**", "events.out.tfevents*"), recursive=True)
    return sorted(set(out))


def _find_report_files():
    """Every sweep report under the module, from either simulator.

    Two writers produce these now: Mujoco/eval_mujoco.py -> mujoco_eval_report_<ckpt>.json and
    unitree_rl_lab/scripts/rsl_rl/sweep.py -> isaac_eval_report_<ckpt>.json. The Isaac ones are
    the primary measurement for any policy whose actor has no base_lin_vel input -- it runs open
    loop on velocity, so its MuJoCo numbers understate it badly (a 1.0 m/s command reads 0.74 in
    MuJoCo and 0.96 in Isaac for the same checkpoint). Match both and let the sidebar say which.
    """
    out = []
    for root in MODULE_DIRS:
        out += glob.glob(os.path.join(root, "**", "*eval_report*.json"), recursive=True)
    return sorted(set(out))


def _reports_by_run_dir():
    """run directory -> [report files]. Reports live in <run>/checkpoints/, so one level up from
    the file is the run the dashboard lists."""
    out = {}
    for f in _find_report_files():
        d = os.path.dirname(f)
        if os.path.basename(d) == "checkpoints":
            d = os.path.dirname(d)
        out.setdefault(d, []).append(f)
    return out


def _canon_tag(tag):
    """skrl >= 2.1.0 logs environment_info keys bare ("reward/alive"); older skrl prefixed them
    ("Info / reward/alive"). Normalise historical runs to the prefixed spelling so runs recorded
    on either machine chart together. New runs are already prefixed at the source, in train.py."""
    if tag.startswith(("reward/", "diag/")):
        return f"Info / {tag}"
    return tag


def _read_scalars(events_path):
    """Cached scalar read. Re-reads only when the file grows, so live runs stay current.
    An eval-only run has no events file at all -- it charts nothing, it is not an error."""
    if not events_path:
        return {}
    try:
        st = os.stat(events_path)
    except OSError:
        return {}
    key = (st.st_mtime, st.st_size)
    with _cache_lock:
        hit = _scalar_cache.get(events_path)
        if hit and hit[0] == key:
            return hit[1]
    if not HAS_TB:
        return {}
    try:
        ea = EventAccumulator(events_path, size_guidance={"scalars": 0})
        ea.Reload()
        data = {_canon_tag(t): [[e.step, e.wall_time, float(e.value)] for e in ea.Scalars(t)]
                for t in ea.Tags()["scalars"]}
    except Exception as e:
        print(f"[viewer] failed reading {events_path}: {e}")
        data = {}
    with _cache_lock:
        _scalar_cache[events_path] = (key, data)
    return data


def _run_index():
    """One entry per training run. A run is normally found by its tfevents file, but a run that
    was evaluated and then copied without its tfevents (or evaluated from checkpoints alone) is
    listed too -- it simply has no curves to draw, and its eval reports stay reachable."""
    runs = []
    now = time.time()
    reports = _reports_by_run_dir()
    seen = set()
    entries = [(os.path.dirname(ev), ev) for ev in _find_event_files()]
    entries += [(d, None) for d in sorted(reports) if d not in {e[0] for e in entries}]
    for run_dir, ev in entries:
        if run_dir in seen:
            continue
        seen.add(run_dir)
        name = os.path.basename(run_dir)
        rel = os.path.relpath(run_dir, BASE_DIR)
        module = ""
        parts = rel.split(os.sep)
        if "IsaacLab_Tasks" in parts:
            i = parts.index("IsaacLab_Tasks")
            if i + 1 < len(parts):
                module = parts[i + 1]
        env_cfg = _load_yaml(os.path.join(run_dir, "params", "env.yaml"))
        agent_cfg = _load_yaml(os.path.join(run_dir, "params", "agent.yaml"))
        cfg = _flatten_cfg(env_cfg, agent_cfg)
        stamped = [ev] if ev else reports.get(run_dir, [])
        mtime = 0
        for f in stamped:
            try:
                mtime = max(mtime, os.stat(f).st_mtime)
            except OSError:
                pass
        runs.append({
            "name": name,
            "id": rel,
            "module": module,
            "events": ev,
            "last_write": mtime,
            "live": bool(ev) and (now - mtime) < LIVE_WINDOW_S,
            "n_reports": len(reports.get(run_dir, [])),
            "num_envs": cfg.get("num_envs"),
            "obs_dim": cfg.get("observation_space"),
            "config": cfg,
        })
    runs.sort(key=lambda r: r["last_write"], reverse=True)
    return runs


PORT = int(os.environ.get("VIEWER_PORT", "8000"))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(BASE_DIR, "Tools", "viewer_frontend")

# Index only this task module. Each IsaacLab_Tasks/<module> is an independent copy of the task with
# its own rewards and its own logs, so indexing all of them charts unrelated reward functions on
# shared axes -- and every extra run costs a tfevents parse on each /api/runs call. Override with
# VIEWER_MODULES=Stairs (etc.) to point the dashboard at another one.
# Comma-separated, because the paper needs the unitree baselines charted against Walk's runs on
# the same axes -- and the alternative was copying reports into Walk/logs by hand, which meant
# two copies of every file and a stale one the moment a sweep was re-run.
#
# The warning above still stands for the general case: two modules with different reward
# functions on one axis is a comparison that has to be intended. VIEWER_MODULES=Walk restores
# the old single-module behaviour.
VIEWER_MODULES = [
    m.strip()
    for m in os.environ.get("VIEWER_MODULES", os.environ.get("VIEWER_MODULE", "Walk,unitree_rl_lab")).split(",")
    if m.strip()
]
MODULE_DIRS = [os.path.join(BASE_DIR, "IsaacLab_Tasks", m) for m in VIEWER_MODULES]
MODULE_DIRS = [d for d in MODULE_DIRS if os.path.isdir(d)]
# Kept for the code paths that still want a single root (run naming, relpath bases).
MODULE_DIR = MODULE_DIRS[0] if MODULE_DIRS else os.path.join(BASE_DIR, "IsaacLab_Tasks", "Walk")

def _report_identity(file_path, meta):
    """Split an eval report into the run it belongs to and the checkpoint inside that run, so the
    viewer can fold the many checkpoints of one training run into a single folder entry.
    Reports live at <run>/checkpoints/mujoco_eval_report_<checkpoint>.json; the checkpoint path in
    the metadata may be a container path (/app/...) so the on-disk path is the reliable source."""
    run_dir = os.path.dirname(file_path)
    if os.path.basename(run_dir) == "checkpoints":
        run_dir = os.path.dirname(run_dir)
    run_name = os.path.basename(run_dir) or "Unknown"
    try:
        run_id = os.path.relpath(run_dir, BASE_DIR)
    except ValueError:
        run_id = run_dir

    label = (meta or {}).get("checkpoint_name") or ""
    if not label or label in ("None", "Unknown"):
        label = re.sub(r"^.*?(mujoco|isaac)_eval_report_?", "", os.path.basename(file_path))
    label = re.sub(r"\.(pt|json)$", "", label)
    label = re.sub(r"^agent_", "", label)
    steps = int(label) if label.isdigit() else None
    if steps is not None:
        label = f"{steps // 1000}k" if steps >= 1000 else str(steps)
    elif label in ("best_agent", "best"):
        label = "best"
    elif not label:
        label = "latest"

    # Which simulator produced it. metadata wins; otherwise read it off the filename, so the
    # older MuJoCo reports (written before the field existed) still identify correctly.
    # Tools/sweep_report.py writes real_eval_report_<ckpt>.json from the robot itself.
    simulator = (meta or {}).get("simulator")
    if not simulator:
        base = os.path.basename(file_path)
        simulator = ("isaac" if "isaac_eval_report" in base
                     else "real" if "real_eval_report" in base else "mujoco")
    # Same checkpoint swept in several places must not collapse into one sidebar entry.
    if simulator in ("isaac", "real"):
        label = f"{label} [{simulator}]"

    return {"run_name": run_name, "run_id": run_id, "simulator": simulator,
            "checkpoint_label": label, "checkpoint_steps": steps}


def _json_safe(obj):
    """Replace NaN/Infinity with null, recursively.

    json.dumps emits bare NaN / Infinity tokens for non-finite floats. That is valid Python and
    invalid JSON, and the browser's JSON.parse rejects the ENTIRE response -- so one report with
    a single NaN in it blanks the whole dashboard with a parse error pointing at a byte offset
    that says nothing about which file is at fault. Sanitising here means a bad report costs its
    own cell, not everyone else's.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class EvalReportHandler(SimpleHTTPRequestHandler):
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        # Dev tool: never let a browser cache the dashboard, or edits appear to do nothing.
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def _json(self, obj):
        body = json.dumps(_json_safe(obj), allow_nan=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The client went away before the response finished -- a reload, a closed tab, or a
            # request the page superseded. Nothing is wrong on this side and there is nobody
            # left to tell, so do not let socketserver print a traceback for it.
            pass

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/api/runs':
            runs = _run_index()
            payload = []
            for r in runs:
                sc = _read_scalars(r["events"])
                steps = 0
                for series in sc.values():
                    if series:
                        steps = max(steps, series[-1][0])
                payload.append({k: v for k, v in r.items() if k != "events"} |
                               {"tags": sorted(sc.keys()), "last_step": steps})
            self._json({"runs": payload, "has_tensorboard": HAS_TB})
            return

        if parsed_path.path == '/api/scalars':
            q = urllib.parse.parse_qs(parsed_path.query)
            wanted = [s for s in q.get("runs", [""])[0].split("|") if s]
            tagfilter = [s for s in q.get("tags", [""])[0].split("|") if s]
            index = {r["id"]: r for r in _run_index()}
            out = {}
            for rid in wanted:
                r = index.get(rid)
                if not r:
                    continue
                sc = _read_scalars(r["events"])
                if tagfilter:
                    sc = {t: v for t, v in sc.items() if t in tagfilter}
                out[rid] = {
                    "series": sc,
                    "num_envs": r["num_envs"],
                    "live": r["live"],
                    "config": r["config"],
                }
            self._json({"runs": out})
            return

        # API endpoint to fetch all reports
        if parsed_path.path == '/api/reports':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            # Handle CORS if needed
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            reports = []
            
            # Reports for the selected task module only, same glob the run index uses.
            all_files = _find_report_files()
            
            for file_path in all_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Ensure it has metadata
                        if "metadata" not in data:
                            # Try to infer it or skip
                            data = {
                                "metadata": {
                                    "checkpoint": "Unknown",
                                    "checkpoint_name": os.path.basename(file_path),
                                    "robot_type": "Unknown",
                                    "timestamp": "Legacy"
                                },
                                "results": data
                            }
                        
                        # Add the file path to metadata for unique identification
                        data["metadata"]["file_path"] = file_path
                        
                        # Tag with run/checkpoint identity so the sidebar can group by run folder,
                        # e.g. run "NiceGait6" + checkpoint "275k" -> "NiceGait6 - 275k".
                        ident = _report_identity(file_path, data["metadata"])
                        data["metadata"].update(ident)
                        data["metadata"]["display_name"] = f"{ident['run_name']} - {ident['checkpoint_label']}"
                        
                        reports.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
            # Sort reports by timestamp, descending
            def get_timestamp(r):
                return r.get("metadata", {}).get("timestamp", "")
                
            reports.sort(key=get_timestamp, reverse=True)
            
            response = json.dumps(_json_safe({"reports": reports}), allow_nan=False)
            try:
                self.wfile.write(response.encode('utf-8'))
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
            
        # Serve frontend files
        return super().do_GET()

def main():
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    
    server_address = ('', PORT)
    # Threading, not the plain HTTPServer. /api/runs parses every tfevents file under every
    # indexed module -- 27 of them once unitree_rl_lab is included -- and on a single-threaded
    # server that blocks the page's own static assets behind it, so the browser gives up
    # mid-response and the log fills with BrokenPipeError tracebacks.
    httpd = ThreadingHTTPServer(server_address, EvalReportHandler)
    
    print("="*60)
    print(f"🚀 Quadruped training / evaluation dashboard running!")
    missing = [m for m in VIEWER_MODULES if not os.path.isdir(os.path.join(BASE_DIR, "IsaacLab_Tasks", m))]
    print(f"📦 Modules    {', '.join(VIEWER_MODULES)}  (set VIEWER_MODULES to change)")
    if missing:
        print(f"⚠️  not found under IsaacLab_Tasks/: {', '.join(missing)} — skipped.")
    print(f"📊 Indexing   {len(_find_report_files())} eval reports, {len(_find_event_files())} runs")
    print(f"🔗 Dashboard  http://localhost:{PORT}/")
    if not HAS_TB:
        print("⚠️  tensorboard not importable — training curves will be empty.")
        print("   Run this under env_isaacsim:  source ~/env_isaacsim/bin/activate")
    print("="*60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    main()
