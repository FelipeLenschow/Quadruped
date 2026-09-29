"""Keepout mask for a saved map: <map>_keepout.pgm/.yaml, same size and origin as the map.

    python3 src/quadruped_description/scripts/make_keepout_mask.py maps/nav --zone -7.5,-4.5,-5.5,-2.5

White is allowed, black is a no-go zone. Zones are rectangles x0,y0,x1,y1 in map metres; or open
the .pgm in an image editor and paint black. nav.launch.py turns the keepout filter on when the
file exists next to the map.
"""

import argparse
import re

import numpy as np

FREE, KEEPOUT = 254, 0


def read_pgm(path):
    data = open(path, "rb").read()
    header = re.match(rb"P5\s+(?:#.*\s+)*(\d+)\s+(\d+)\s+(\d+)\s", data)
    w, h = int(header.group(1)), int(header.group(2))
    return np.frombuffer(data[header.end():header.end() + w * h], np.uint8).reshape(h, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("map", help="map path without extension, e.g. maps/nav")
    ap.add_argument("--zone", action="append", default=[], help="x0,y0,x1,y1 in metres (repeatable)")
    args = ap.parse_args()

    meta = dict(line.split(":", 1) for line in open(args.map + ".yaml") if ":" in line)
    res = float(meta["resolution"])
    ox, oy = [float(v) for v in meta["origin"].strip(" []\n").split(",")[:2]]
    h, w = read_pgm(args.map + ".pgm").shape
    mask = np.full((h, w), FREE, np.uint8)
    for zone in args.zone:
        x0, y0, x1, y1 = [float(v) for v in zone.split(",")]
        c0, c1 = sorted((int((x0 - ox) / res), int((x1 - ox) / res)))
        r0, r1 = sorted((h - int((y0 - oy) / res), h - int((y1 - oy) / res)))
        mask[max(r0, 0):min(r1, h), max(c0, 0):min(c1, w)] = KEEPOUT

    out = args.map + "_keepout"
    with open(out + ".pgm", "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(mask.tobytes())
    with open(out + ".yaml", "w") as f:
        f.write(f"image: {out.rsplit('/', 1)[-1]}.pgm\nmode: trinary\nresolution: {res}\n"
                f"origin: [{ox}, {oy}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    print(f"{out}.pgm/.yaml: {len(args.zone)} zone(s)")


if __name__ == "__main__":
    main()
