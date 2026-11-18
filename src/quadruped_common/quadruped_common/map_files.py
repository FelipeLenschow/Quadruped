"""Nav2 map files from an OccupancyGrid, with no ROS: map_save writes them from /map, Tools/map_video.py from an MCAP.

The same files as nav2's map_saver_cli (trinary: free, occupied, unknown), so launcher [N] lists the map and AMCL
can localize on it. The grid only needs .info (width, height, resolution, origin) and .data, so a message decoded by
mcap_ros2 also works.
"""

import math
import os

import numpy as np

# map_server reads a pixel as free below free_thresh: 0.196 keeps unknown's grey 205 ((255 - 205) / 255) unknown.
FREE_THRESH, OCCUPIED_THRESH = 0.196, 0.65
FREE, OCCUPIED, UNKNOWN = 254, 0, 205


def origin_yaw(info):
    q = info.origin.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def trinary(cells):
    """cells: int8 occupancy (-1 unknown, 0 to 100), row 0 at the origin. Returns uint8 FREE / OCCUPIED / UNKNOWN."""
    image = np.full(cells.shape, UNKNOWN, np.uint8)
    image[(cells >= 0) & (cells <= 25)] = FREE
    image[cells >= OCCUPIED_THRESH * 100] = OCCUPIED
    return image


def write_map(grid, base):
    """grid: OccupancyGrid. Writes base.pgm and base.yaml."""
    info = grid.info
    image = trinary(np.asarray(grid.data, np.int8).reshape(info.height, info.width))
    with open(base + ".pgm", "wb") as f:
        f.write(f"P5\n{info.width} {info.height}\n255\n".encode())
        f.write(np.flipud(image).tobytes())
    o = info.origin.position
    with open(base + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(base)}.pgm\nmode: trinary\nresolution: {info.resolution:.6g}\n"
                f"origin: [{o.x:.6g}, {o.y:.6g}, {origin_yaw(info):.6g}]\nnegate: 0\n"
                f"occupied_thresh: {OCCUPIED_THRESH}\nfree_thresh: {FREE_THRESH}\n")
