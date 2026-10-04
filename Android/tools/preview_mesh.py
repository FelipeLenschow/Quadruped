"""Software z-buffer render of the exported Go2 meshes at the stand pose, lit like the app's
shader, to check an export without a phone:

    unset PYTHONPATH; ~/env_isaacsim/bin/python Android/tools/preview_mesh.py out.png
"""
import struct, numpy as np, xml.etree.ElementTree as ET
from PIL import Image
import sys
from pathlib import Path
A = str(Path(__file__).resolve().parents[1] / "app/src/main/assets/go2") + "/"
def rot(axis, a):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis); x, y, z = axis; c, s = np.cos(a), np.sin(a); C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],[y*x*C+z*s, c+y*y*C, y*z*C-x*s],[z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
def origin(e):
    T = np.eye(4)
    if e is None: return T
    xyz = [float(v) for v in e.get("xyz", "0 0 0").split()]; r, p, y = [float(v) for v in e.get("rpy", "0 0 0").split()]
    T[:3,:3] = rot([0,0,1], y) @ rot([0,1,0], p) @ rot([1,0,0], r); T[:3,3] = xyz; return T
def mesh(name):
    b = open(A + name + ".mesh", "rb").read(); o = 8; parts = []
    for _ in range(struct.unpack_from("<I", b, 4)[0]):
        col = struct.unpack_from("<4f", b, o); o += 16; nv, ni = struct.unpack_from("<II", b, o); o += 8
        vn = np.frombuffer(b, "<f4", nv*6, o).reshape(-1, 6); o += nv*24
        f = np.frombuffer(b, "<u4", ni, o).reshape(-1, 3); o += ni*4; parts.append((np.array(col), vn[:, :3], vn[:, 3:], f))
    return parts
root = ET.parse(A + "go2.urdf").getroot()
q = {}
for l in ("FL", "FR", "RL", "RR"):
    q[l+"_hip_joint"] = 0.1 if l[1] == "L" else -0.1; q[l+"_thigh_joint"] = 0.8 if l[0] == "F" else 1.0; q[l+"_calf_joint"] = -1.5
poses = {"base": np.eye(4)}
for _ in range(10):
    for j in root.findall("joint"):
        par, ch = j.find("parent").get("link"), j.find("child").get("link")
        if par in poses and ch not in poses:
            M = np.eye(4)
            if j.get("type") in ("revolute", "continuous"):
                M[:3,:3] = rot([float(v) for v in j.find("axis").get("xyz").split()], q.get(j.get("name"), 0.0))
            poses[ch] = poses[par] @ origin(j.find("origin")) @ M
P, N, C = [], [], []
for link in root.findall("link"):
    for vis in link.findall("visual"):
        m = vis.find("geometry/mesh")
        if m is None: continue
        T = poses[link.get("name")] @ origin(vis.find("origin"))
        for col, v, n, f in mesh(m.get("filename").split("/")[-1][:-4]):
            P.append(((T[:3,:3] @ v.T).T + T[:3,3])[f]); N.append((T[:3,:3] @ n.T).T[f]); C += [col[:3]] * len(f)
P = np.concatenate(P); N = np.concatenate(N); C = np.array(C)
W, H = 640, 480
yaw, pitch, dist = -2.3, 0.4, 1.4
eye = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)]) * dist
fwd = -eye / np.linalg.norm(eye); right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
focal = H / 2 / np.tan(np.radians(38) / 2)
d = P - eye; z = d @ fwd; sx = W/2 + focal * (d @ right) / z; sy = H/2 - focal * (d @ up) / z
img = np.full((H, W, 3), [0x0B, 0x0D, 0x12], float) / 255; zb = np.full((H, W), np.inf)
L = np.array([0.36, 0.48, 0.80]); L /= np.linalg.norm(L)
for t in range(len(P)):
    xs, ys, zs = sx[t], sy[t], z[t]
    x0, x1 = int(max(0, np.floor(xs.min()))), int(min(W-1, np.ceil(xs.max()))); y0, y1 = int(max(0, np.floor(ys.min()))), int(min(H-1, np.ceil(ys.max())))
    if x0 > x1 or y0 > y1: continue
    gx, gy = np.meshgrid(np.arange(x0, x1+1) + 0.5, np.arange(y0, y1+1) + 0.5)
    (ax, bx, cx), (ay, by, cy) = xs, ys
    den = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy)
    if abs(den) < 1e-9: continue
    w0 = ((by-cy)*(gx-cx) + (cx-bx)*(gy-cy)) / den; w1 = ((cy-ay)*(gx-cx) + (ax-cx)*(gy-cy)) / den; w2 = 1 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any(): continue
    depth = w0*zs[0] + w1*zs[1] + w2*zs[2]
    sub = zb[y0:y1+1, x0:x1+1]; win = inside & (depth < sub)
    if not win.any(): continue
    sub[win] = depth[win]
    n = N[t].mean(0); n /= np.linalg.norm(n) + 1e-9
    v = -fwd
    if n @ v < 0: n = -n
    alb = np.maximum(C[t], 0.11); dif = max(n @ L, 0); hv = (L + v) / np.linalg.norm(L + v)
    col = alb * (0.3 + 0.7*dif) + 0.12 * max(n @ hv, 0)**40 + np.array([0.24, 0.84, 0.78]) * 0.22 * (1 - max(n @ v, 0))**3
    img[y0:y1+1, x0:x1+1][win] = np.clip(col, 0, 1)
out = sys.argv[1] if len(sys.argv) > 1 else "go2_preview.png"
Image.fromarray((img*255).astype(np.uint8)).save(out)
print(out)
