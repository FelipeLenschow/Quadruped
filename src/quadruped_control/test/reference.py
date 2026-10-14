"""Recorded outputs of the Python control core (quadruped_core's pipeline, before the C++ port) that the parity tests replay.

A reference is an .npz of arrays plus a JSON "meta" entry. Meta carries what the recording depended on: the seed,
a hash of the generated inputs, the config files (written back to a temporary directory on replay) and the
checkpoints by hash, so it stays valid when config.yaml or policies.json change later.
"""
import contextlib
import hashlib
import json
import os
import tempfile

import numpy as np

from quadruped_common import paths

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "reference")


def path(name):
    return os.path.join(DIR, name)


def save(file, meta, **arrays):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    np.savez_compressed(file, meta=np.array(json.dumps(meta)), **arrays)


def load(file):
    with np.load(file, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return json.loads(str(arrays.pop("meta"))), arrays


class Hash:
    """sha1 over a stream of inputs: arrays by their bytes, anything else by repr."""

    def __init__(self):
        self.h = hashlib.sha1()

    def add(self, *values):
        for v in values:
            if isinstance(v, np.ndarray):
                self.h.update(str(v.dtype).encode() + v.tobytes())
            elif isinstance(v, dict):
                for k in sorted(v):
                    self.add(k, v[k])
            elif isinstance(v, (list, tuple)):
                self.h.update(b"[")
                self.add(*v)
                self.h.update(b"]")
            else:
                self.h.update(repr(v).encode())
        return self

    def hexdigest(self):
        return self.h.hexdigest()


def file_sha1(file):
    h = hashlib.sha1()
    with open(file, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_files(names):
    """{name: text} of config files, by name: config.yaml, policies.json, go2_kinematics.yaml."""
    where = {"config.yaml": paths.config("config.yaml"), "policies.json": paths.config("policies.json"),
             "go2_kinematics.yaml": paths.description("config", "go2_kinematics.yaml")}
    out = {}
    for name in names:
        with open(where[name]) as f:
            out[name] = f.read()
    return out


@contextlib.contextmanager
def frozen(files):
    """The recorded config files in a temporary directory, with quadruped_common.paths pointing at them."""
    config, description = paths.config, paths.description
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in files.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(text)

        def pick(original, *parts):
            local = os.path.join(tmp, parts[-1]) if parts else None
            return local if local and parts[-1] in files else original(*parts)

        paths.config = lambda *parts: pick(config, *parts)
        paths.description = lambda *parts: pick(description, *parts)
        try:
            yield tmp
        finally:
            paths.config, paths.description = config, description


def checkpoints(catalog):
    """{relative path: sha1} of the catalog's checkpoints that exist."""
    out = {}
    for spec in catalog.values():
        p = spec.get("path") or ""
        full = p if os.path.isabs(p) else paths.repo(p)
        if p and os.path.isfile(full):
            out[p] = file_sha1(full)
    return out


def find_checkpoint(rel, sha1, live_catalog=None):
    """The recorded checkpoint: at its recorded path, or wherever policies.json now puts a file with that hash."""
    candidates = [rel] + [s.get("path") or "" for s in (live_catalog or {}).values()]
    for p in candidates:
        full = p if os.path.isabs(p) else paths.repo(p)
        if p and os.path.isfile(full) and file_sha1(full) == sha1:
            return full
    return None
