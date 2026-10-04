"""Export the Go2 URDF and its meshes for the app's 3D view.

    unset PYTHONPATH; ~/env_isaacsim/bin/python Android/tools/export_urdf.py

Copies the URDF to app/src/main/assets/go2/ and turns each DAE it references into a
.mesh file: the DAE's node transforms baked in, one part per material colour,
decimated so the whole robot stays light enough for a phone.

.mesh (little-endian): b"QMSH", u32 parts; per part: f32 rgba[4], u32 n_vertices,
u32 n_indices, f32 [x y z nx ny nz] * n_vertices, u32 indices.
"""

import re
import shutil
import struct
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "src/quadruped_description/robots/Unitree_Go2/models/go2_description"
URDF = PKG / "urdf/go2_description.urdf"
OUT = REPO / "Android/app/src/main/assets/go2"
# Triangles kept per mesh file (before instancing: hip and foot are used four times).
BUDGET = {"base": 28000, "hip": 4500, "thigh": 6500, "thigh_mirror": 6500,
          "calf": 4000, "calf_mirror": 4000, "foot": 800}


def parts(path: Path, budget: int):
    scene = trimesh.load(path, force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        g = scene.geometry[geom_name].copy()
        g.apply_transform(transform)
        meshes.append(g)
    total = sum(len(m.faces) for m in meshes)
    out = []
    for m in meshes:
        color = np.asarray(m.visual.material.main_color, dtype=np.float32) / 255.0
        # The DAEs give every face its own vertices; decimating that soup tears holes
        # between faces, so weld by position first.
        welded = trimesh.Trimesh(m.vertices, m.faces, process=True)
        v, f = np.asarray(welded.vertices, np.float32), np.asarray(welded.faces, np.int64)
        keep = max(12, int(budget * len(f) / total))
        if len(f) > keep:
            v, f = fast_simplification.simplify(v, f, target_reduction=1.0 - keep / len(f))
        t = trimesh.Trimesh(v, f, process=True)
        t.update_faces(t.nondegenerate_faces())
        t.remove_unreferenced_vertices()
        # Split vertices across creases: averaging over a thin shell's two sides gives
        # a zero normal, which the shader turns into NaN and the triangle vanishes.
        t = trimesh.graph.smooth_shade(t, angle=np.radians(35))
        n = np.asarray(t.vertex_normals, np.float32)
        out.append((color, np.asarray(t.vertices, np.float32), n, np.asarray(t.faces, np.uint32)))
    return out


def open_edges(v, faces):
    """Edges used by one face only, after merging smooth_shade's split vertices."""
    faces = trimesh.Trimesh(v, faces, process=True).faces
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())


def write(path: Path, mesh_parts):
    with open(path, "wb") as fh:
        fh.write(b"QMSH" + struct.pack("<I", len(mesh_parts)))
        for color, v, n, f in mesh_parts:
            fh.write(struct.pack("<4f", *color))
            fh.write(struct.pack("<II", len(v), f.size))
            fh.write(np.hstack([v, n]).astype("<f4").tobytes())
            fh.write(f.astype("<u4").tobytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = URDF.read_text()
    shutil.copy(URDF, OUT / "go2.urdf")
    names = sorted(set(re.findall(r'package://go2_description/dae/(\w+)\.dae', text)))
    for name in names:
        mesh_parts = parts(PKG / "dae" / f"{name}.dae", BUDGET.get(name, 3000))
        write(OUT / f"{name}.mesh", mesh_parts)
        lo = np.min([p[1].min(0) for p in mesh_parts], 0)
        hi = np.max([p[1].max(0) for p in mesh_parts], 0)
        tris = sum(p[3].size // 3 for p in mesh_parts)
        holes = sum(open_edges(p[1], p[3]) for p in mesh_parts)
        print(f"{name}: {len(mesh_parts)} parts, {tris} tris, {holes} open edges, size {np.round(hi - lo, 3)}")


if __name__ == "__main__":
    main()
