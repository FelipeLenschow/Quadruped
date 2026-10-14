"""policies.json entries and checkpoints, in the form the C++ Controller takes them."""
import os
import tempfile

from ._core import PolicyEntry, PolicySpec
from .export import read_checkpoint, run_params, write_qpol

PROTECTIONS = ("heartbeat", "tilt", "pitch", "rom")


def read_catalog(policies, warn=print):
    """{name: {"path", "protections", "commands"}} as PolicyManager._read_catalog reads policies.json."""
    out = {}
    for name, spec in (policies or {}).items():
        spec = spec or {}
        protections = spec.get("protections")
        protections = PROTECTIONS if protections is None else tuple(protections)
        unknown = [p for p in protections if p not in PROTECTIONS]
        if unknown:
            warn(f"[Policies] Policy '{name}': unknown protections {unknown}, known: {list(PROTECTIONS)}. Using all of them.")
            protections = PROTECTIONS
        commands = spec.get("commands", True)
        if commands not in (True, False, "trick"):
            warn(f"[Policies] Policy '{name}': commands {commands!r} is not true, false or \"trick\". Using true.")
            commands = True
        out[name] = {"path": spec.get("path", ""), "protections": protections, "commands": commands}
    return out


def entry_for(spec, checkpoint=None):
    """The PolicyEntry of a catalog spec; a trick reads its clock and switches from the run's env.yaml."""
    clock_hz, switches = 0.0, 0
    if spec["commands"] == "trick" and checkpoint:
        env = run_params(checkpoint, "env.yaml") or {}
        trick = (env.get("commands") or {}).get("trick") or {}
        clock_hz = float(trick.get("clock_hz") or 0.0)
        switches = int(trick.get("switches") or 1)
    return PolicyEntry(protections=list(spec["protections"]), commands=spec["commands"], clock_hz=clock_hz,
                       switches=switches)


def load_spec(checkpoint, name=""):
    """PolicySpec for a checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "policy.qpol")
        write_qpol(path, *read_checkpoint(checkpoint, name=name))
        return PolicySpec.from_file(path)
