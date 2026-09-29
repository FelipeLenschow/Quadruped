"""Rasterize a Gazebo world's static boxes, cylinders and cones into a Nav2 map (.pgm + .yaml).

    python3 src/quadruped_description/scripts/sdf_to_map.py src/quadruped_description/gazebo/nav.sdf maps/nav

Inside the walls is free, the obstacles occupied, outside unknown. The map frame is the world frame.
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

CONE_RADIUS = 0.18   # the traffic_cone model's 0.36 m base
FREE, OCCUPIED, UNKNOWN = 254, 0, 205


def shapes(world_file):
    """(x, y, yaw, kind, size) for each static model; kind box (size = sx, sy) or circle (size = r)."""
    world = ET.parse(world_file).getroot().find("world")
    out = []
    for model in world.findall("model"):
        pose = [float(v) for v in (model.findtext("pose") or "0 0 0 0 0 0").split()]
        geom = model.find("link/collision/geometry")
        if geom is None or model.findtext("static", "false").strip() != "true":
            continue
        if geom.find("box") is not None:
            sx, sy, _ = [float(v) for v in geom.findtext("box/size").split()]
            out.append((pose[0], pose[1], pose[5], "box", (sx, sy)))
        elif geom.find("cylinder") is not None:
            out.append((pose[0], pose[1], 0.0, "circle", float(geom.findtext("cylinder/radius"))))
    for inc in world.findall("include"):
        if "traffic_cone" in (inc.findtext("uri") or ""):
            pose = [float(v) for v in inc.findtext("pose").split()]
            out.append((pose[0], pose[1], 0.0, "circle", CONE_RADIUS))
    return out


def distance(shape, px, py):
    """Distance from the points (px, py) to the shape's edge, 0 inside."""
    x, y, yaw, kind, size = shape
    dx, dy = px - x, py - y
    if kind == "circle":
        return np.maximum(np.hypot(dx, dy) - size, 0.0)
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return np.hypot(np.maximum(np.abs(lx) - size[0] / 2, 0.0), np.maximum(np.abs(ly) - size[1] / 2, 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("world")
    ap.add_argument("out", help="output path without extension, e.g. maps/nav")
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.5)
    args = ap.parse_args()

    items = shapes(args.world)
    walls = [s for s in items if s[3] == "box" and max(s[4]) > 5.0]
    xs = [s[0] for s in walls]
    ys = [s[1] for s in walls]
    x0, x1 = min(xs) - args.margin, max(xs) + args.margin
    y0, y1 = min(ys) - args.margin, max(ys) + args.margin
    res = args.resolution
    w, h = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    px = x0 + (np.arange(w) + 0.5) * res
    py = y1 - (np.arange(h) + 0.5) * res
    PX, PY = np.meshgrid(px, py)

    image = np.full((h, w), UNKNOWN, np.uint8)
    inside = (PX > min(xs)) & (PX < max(xs)) & (PY > min(ys)) & (PY < max(ys))
    image[inside] = FREE
    for shape in items:
        image[distance(shape, PX, PY) < res / 2] = OCCUPIED

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".pgm", "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(image.tobytes())
    with open(args.out + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(args.out)}.pgm\nmode: trinary\nresolution: {res}\n"
                f"origin: [{x0:.3f}, {y0:.3f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    print(f"{args.out}.pgm/.yaml: {w}x{h} px, {len(items)} shapes, origin ({x0:.2f}, {y0:.2f})")


if __name__ == "__main__":
    main()
