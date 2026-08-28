import os
import json
import glob
from http.server import SimpleHTTPRequestHandler, HTTPServer
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
    pat = os.path.join(MODULE_DIR, "**", "events.out.tfevents*")
    return sorted(set(glob.glob(pat, recursive=True)))


def _canon_tag(tag):
    """skrl >= 2.1.0 logs environment_info keys bare ("reward/alive"); older skrl prefixed them
    ("Info / reward/alive"). Normalise historical runs to the prefixed spelling so runs recorded
    on either machine chart together. New runs are already prefixed at the source, in train.py."""
    if tag.startswith(("reward/", "diag/")):
        return f"Info / {tag}"
    return tag


def _read_scalars(events_path):
    """Cached scalar read. Re-reads only when the file grows, so live runs stay current."""
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
    runs = []
    now = time.time()
    for ev in _find_event_files():
        run_dir = os.path.dirname(ev)
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
        try:
            mtime = os.stat(ev).st_mtime
        except OSError:
            mtime = 0
        runs.append({
            "name": name,
            "id": rel,
            "module": module,
            "events": ev,
            "last_write": mtime,
            "live": (now - mtime) < LIVE_WINDOW_S,
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
# VIEWER_MODULE=Stairs (etc.) to point the dashboard at another one.
VIEWER_MODULE = os.environ.get("VIEWER_MODULE", "Walk")
MODULE_DIR = os.path.join(BASE_DIR, "IsaacLab_Tasks", VIEWER_MODULE)

class EvalReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        # Dev tool: never let a browser cache the dashboard, or edits appear to do nothing.
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def _json(self, obj):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
            
            # Search for mujoco_eval_report*.json under the selected task module only.
            search_pattern = os.path.join(MODULE_DIR, "**", "*mujoco_eval_report*.json")
            all_files = sorted(set(glob.glob(search_pattern, recursive=True)))
            
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
                        
                        # Generate a nice display name (e.g., "NiceGait6 - 275k")
                        checkpoint_path = data["metadata"].get("checkpoint", "")
                        if checkpoint_path and "checkpoints" in checkpoint_path:
                            parts = checkpoint_path.split(os.sep)
                            try:
                                chk_idx = parts.index("checkpoints")
                                run_name = parts[chk_idx - 1]
                                agent_name = parts[-1].replace(".pt", "").replace("agent_", "")
                                if agent_name.isdigit():
                                    steps = int(agent_name)
                                    if steps >= 1000:
                                        agent_name = f"{steps//1000}k"
                                data["metadata"]["display_name"] = f"{run_name} - {agent_name}"
                            except Exception:
                                data["metadata"]["display_name"] = data["metadata"].get("checkpoint_name", "Unknown")
                        else:
                            data["metadata"]["display_name"] = data["metadata"].get("checkpoint_name", "Unknown")
                        
                        reports.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
            # Sort reports by timestamp, descending
            def get_timestamp(r):
                return r.get("metadata", {}).get("timestamp", "")
                
            reports.sort(key=get_timestamp, reverse=True)
            
            response = json.dumps({"reports": reports})
            self.wfile.write(response.encode('utf-8'))
            return
            
        # Serve frontend files
        return super().do_GET()

def main():
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, EvalReportHandler)
    
    print("="*60)
    print(f"🚀 Policy Evaluation Dashboard running!")
    print(f"📦 Module      {VIEWER_MODULE}  (set VIEWER_MODULE to change)")
    print(f"🔗 Evaluation  http://localhost:{PORT}/")
    print(f"📈 Training    http://localhost:{PORT}/training.html")
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
