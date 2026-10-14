#!/usr/bin/env python3
"""Export checkpoints (paths, or names from policies.json; none = all of them) to .qpol files."""
import argparse
import os
import sys

from quadruped_common import paths
from quadruped_common.config_loader import load_policies
from quadruped_control.export import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policies", nargs="*", help="checkpoint paths or policies.json names")
    parser.add_argument("--out", help="output folder (default: beside each checkpoint)")
    args = parser.parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    catalog = (load_policies().get("policies") or {})
    wanted = args.policies or list(catalog)
    failed = 0
    for item in wanted:
        name, path = (item, catalog[item]["path"]) if item in catalog else (os.path.splitext(os.path.basename(item))[0], item)
        path = path if os.path.isabs(path) else paths.repo(path)
        out = os.path.join(args.out, f"{name}.qpol") if args.out else os.path.splitext(path)[0] + ".qpol"
        try:
            meta = export_checkpoint(path, out, name=name)
            print(f"{name:<12} {meta['layout']:<14} obs {meta['obs_dim']:<5} -> {out}")
        except Exception as e:
            failed += 1
            print(f"{name:<12} FAILED: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
