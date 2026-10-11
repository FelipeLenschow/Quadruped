"""The Go2's four rubber belly pads, which the collision model leaves out.

Each pad is a convex hull of its part of the body's visual mesh (mujoco_menagerie
unitree_go2/assets/base_*.obj, base frame), cut to 48 vertices and stored in go2_belly_pads.json;
`python go2_belly_pads.py <mesh dir>` rebuilds that file. Rear pads sit at x -0.109, y +-0.046 and reach
z -0.077, front pads at x +0.105, y +-0.046 reach z -0.096; the flat belly between them is at z -0.057
and the chin (lidar cage) at x +0.298 reaches z -0.097. Resting on the pads tips the body ~5 deg nose up
and keeps the chin ~1.6 cm off the ground. Without them the chin collider is the lowest point of the
body, so the robot lies on its face.

The chin collider is also moved up 1 cm: it is a 4.7 cm sphere whose bottom sits at z -0.107, 1 cm below
the chin in the mesh.
"""

import json
import os

HULLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2_belly_pads.json")
CHIN_RAISE = 0.010
# Mesh points below the flat belly (z -0.057) inside these boxes, in the base frame: (x range, sign of y).
PAD_REGIONS = {
    "rear_left": ((-0.135, -0.085), 1),
    "rear_right": ((-0.135, -0.085), -1),
    "front_left": ((0.08, 0.135), 1),
    "front_right": ((0.08, 0.135), -1),
}
HULL_VERTICES = 48


def _load_hulls() -> dict:
    with open(HULLS_FILE) as f:
        return json.load(f)


def _spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    spawn = getattr(spawn_from_usd, "__wrapped__", spawn_from_usd)
    prim = spawn(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    for name, hull in _load_hulls().items():
        pad = UsdGeom.Mesh.Define(stage, f"{prim_path}/base/belly_pad_{name}")
        pad.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in hull["points"]]))
        pad.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull["faces"])))
        pad.CreateFaceVertexIndicesAttr(Vt.IntArray([i for face in hull["faces"] for i in face]))
        pad.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(pad.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    raised = 0
    for child in Usd.PrimRange(stage.GetPrimAtPath(f"{prim_path}/Head_lower")):
        if child.IsA(UsdGeom.Sphere) and child.HasAPI(UsdPhysics.CollisionAPI):
            ops = UsdGeom.Xformable(child).GetOrderedXformOps()
            translate = next((o for o in ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            if translate is None:
                translate = UsdGeom.Xformable(child).AddTranslateOp()
                translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
            t = translate.Get()
            translate.Set(type(t)(t[0], t[1], t[2] + CHIN_RAISE))
            raised += 1
    if raised != 1:
        raise RuntimeError(f"expected one chin collider sphere under {prim_path}/Head_lower, found {raised}")
    return prim


def spawn_go2_with_belly_pads(prim_path: str, cfg, *args, **kwargs):
    """UsdFileCfg spawner: the Go2 USD, plus the pads and the corrected chin."""
    from isaaclab.sim.utils import clone

    return clone(_spawn)(prim_path, cfg, *args, **kwargs)


def build_hulls(mesh_dir: str) -> dict:
    """Cut each pad out of the body mesh and reduce its convex hull to HULL_VERTICES points."""
    import numpy as np
    import trimesh

    points = np.concatenate([trimesh.load(os.path.join(mesh_dir, f"base_{i}.obj"), force="mesh").vertices for i in range(5)])
    hulls = {}
    for name, ((x_lo, x_hi), side) in PAD_REGIONS.items():
        y = points[:, 1] * side
        pad = points[(points[:, 0] > x_lo) & (points[:, 0] < x_hi) & (points[:, 2] < -0.058) & (y > 0.02) & (y < 0.075)]
        # Close the top inside the body so the hull is a solid bump, not a shell open at the belly.
        top = pad.copy()
        top[:, 2] = -0.05
        hull = trimesh.convex.convex_hull(np.concatenate([pad, top]))
        keep = [int(np.argmin(hull.vertices[:, 2]))]
        dist = np.linalg.norm(hull.vertices - hull.vertices[keep[0]], axis=1)
        while len(keep) < min(HULL_VERTICES, len(hull.vertices)):
            keep.append(int(np.argmax(dist)))
            dist = np.minimum(dist, np.linalg.norm(hull.vertices - hull.vertices[keep[-1]], axis=1))
        reduced = trimesh.convex.convex_hull(hull.vertices[keep])
        hulls[name] = {"points": reduced.vertices.round(5).tolist(), "faces": reduced.faces.tolist()}
    return hulls


if __name__ == "__main__":
    import sys

    hulls = build_hulls(sys.argv[1])
    with open(HULLS_FILE, "w") as f:
        json.dump(hulls, f)
    for name, hull in hulls.items():
        z = [p[2] for p in hull["points"]]
        print(f"{name}: {len(hull['points'])} vertices, {len(hull['faces'])} faces, lowest z {min(z):+.4f}")
