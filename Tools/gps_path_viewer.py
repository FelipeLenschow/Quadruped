#!/usr/bin/env python3
"""The robot's path from the LIO next to the GPS fixes, with each fix's reported error, over the map.

    python3 Tools/gps_path_viewer.py Mcap/Recordings/out2 --pcd ~/Desktop/Save/Maps_2026-10-23/fastlio_deskewed.pcd
    python3 Tools/gps_path_viewer.py Mcap/Recordings/out2 --png path.png --no-3d   # a top-down image instead

Reads the bag (ROS 2 MCAP): the LIO's odometry (--odom, FAST-LIO's /helios_lio/odom by default), /gps/fix and
gps_driver's /gps/status. The map is optional: a PCD in the same frame as the odometry (camera_init for FAST-LIO,
e.g. lio_cloud_save's or one built from the bag).

The GPS is in latitude, longitude, the LIO in its own frame from where it started: the fixes are put in a local
east-north-up frame, and turned and moved onto the LIO's path (a rotation about z, a shift; least squares on
the first --fit_share of the RTK fixed fixes, weighted by their accuracy). Fitted at the start only, the LIO's
drift is not spread over the run: it shows as an error growing from the start. The GPS measures the antenna (--antenna, its mount on the base): the
two are compared at the antenna, and each fix is drawn moved back to the base with the LIO's attitude at that
time, so both paths are the base's. What is left between them is GPS error plus LIO drift.

Each fix is a dot with a circle of --sigma times its reported horizontal accuracy (1 sigma by default), coloured
by its solution: RTK fixed green, RTK float amber, DGNSS orange, plain GNSS red. The base from the SLAM (LIO) is
white, from the GPS magenta. The window's panel colours the map (by height or intensity, with a colour scale), sets
its point size and opacity, the GPS dots' size and the circles' sigma, and shows or hides each layer. Mouse, as
pcl_viewer, with the horizon kept level: left-drag turns about the view's centre, middle-drag (or shift+left) pans,
right-drag and the scroll wheel zoom.

--color intensity colours the map by the lidar's return strength instead of height, by rank (each colour the same
share of the points). Each lidar has its own scale (the Helios' reflectivity is mostly 10-80, the L1's 150-250), so
in a map from both lidars the L1's points take the top of the scale.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation, Slerp

STYLE = {   # solution -> RGB
    "RTK fixed": (0.20, 0.90, 0.35),
    "RTK float": (0.98, 0.72, 0.20),
    "DGNSS": (1.00, 0.45, 0.15),
    "GNSS fix": (0.95, 0.20, 0.25),
}
BG_COLOR = np.array([0.08, 0.08, 0.12])
GPS_LINE = (1.0, 0.35, 0.85)   # the base path from the GPS
WGS84_A, WGS84_E2 = 6378137.0, 6.69437999014e-3


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def read_bag(path, odom_topic):
    files = sorted(glob.glob(os.path.join(path, "*.mcap"))) if os.path.isdir(path) else [path]
    odom, fixes, status = [], [], []
    for f in files:
        with open(f, "rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _, channel, message, m in reader.iter_decoded_messages(topics=[odom_topic, "/gps/fix", "/gps/status"]):
                if channel.topic == odom_topic:
                    p, q = m.pose.pose.position, m.pose.pose.orientation
                    odom.append((stamp(m.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                elif channel.topic == "/gps/fix":
                    c = m.position_covariance
                    fixes.append((stamp(m.header), message.log_time * 1e-9, m.status.status, m.latitude, m.longitude,
                                  m.altitude, np.sqrt(max(c[0], 0.0)), np.sqrt(max(c[8], 0.0))))
                else:
                    status.append((message.log_time * 1e-9, json.loads(m.data).get("fix", "")))
    if not odom:
        sys.exit(f"no {odom_topic} in {path}")
    if not fixes:
        sys.exit(f"no /gps/fix in {path}")
    odom = np.array(sorted(odom))
    odom = odom[np.concatenate([[True], np.diff(odom[:, 0]) > 0])]
    return odom, fixes, status


def solution(fix, status):
    """The fix's solution: gps_driver's /gps/status at that time, or from NavSatFix (status 2 is RTK: fixed if under
    5 cm)."""
    if status:
        t = np.array([s[0] for s in status])
        i = int(np.argmin(abs(t - fix[1])))
        if abs(t[i] - fix[1]) < 0.6 and status[i][1] in STYLE:
            return status[i][1]
    return {2: "RTK fixed" if fix[6] < 0.05 else "RTK float", 1: "DGNSS", 0: "GNSS fix"}.get(fix[2])


def enu(lat, lon, alt, origin):
    def ecef(la, lo, h):
        la, lo = np.radians(la), np.radians(lo)
        n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
        return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                         (n * (1 - WGS84_E2) + h) * np.sin(la)], -1)
    la0, lo0 = np.radians(origin[0]), np.radians(origin[1])
    d = ecef(lat, lon, alt) - ecef(*origin)
    r = np.array([[-np.sin(lo0), np.cos(lo0), 0],
                  [-np.sin(la0) * np.cos(lo0), -np.sin(la0) * np.sin(lo0), np.cos(la0)],
                  [np.cos(la0) * np.cos(lo0), np.cos(la0) * np.sin(lo0), np.sin(la0)]])
    return d @ r.T


def base_poses(odom, t):
    """The LIO's base position and rotation at times t (inside the odometry's span)."""
    rot = Slerp(odom[:, 0], Rotation.from_quat(odom[:, 4:8]))(t)
    return np.stack([np.interp(t, odom[:, 0], odom[:, k]) for k in (1, 2, 3)], 1), rot


def align(src, dst, w):
    """Weighted 2D rotation and shift (and a z shift) taking src onto dst."""
    ws = w / w.sum()
    ms, md = ws @ src[:, :2], ws @ dst[:, :2]
    h = ((src[:, :2] - ms) * ws[:, None]).T @ (dst[:, :2] - md)
    u, _, vt = np.linalg.svd(h)
    d = np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    r = vt.T @ d @ u.T
    t = md - r @ ms
    dz = ws @ (dst[:, 2] - src[:, 2])
    return lambda p: np.column_stack([(p[:, :2] @ r.T) + t, p[:, 2] + dz]), np.degrees(np.arctan2(r[1, 0], r[0, 0]))


def load_map(path, voxel):
    """The PCD's points and its intensity field (None without one), averaged per voxel."""
    import open3d as o3d
    t = o3d.t.io.read_point_cloud(path)
    points = t.point.positions.numpy().astype(np.float64)
    intensity = t.point["intensity"].numpy().reshape(-1).astype(np.float64) if "intensity" in t.point else None
    keep = np.isfinite(points).all(axis=1)
    points = points[keep]
    intensity = intensity[keep] if intensity is not None else None
    if voxel > 0 and len(points):
        _, inverse, counts = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_inverse=True,
                                       return_counts=True)
        inverse = inverse.reshape(-1)
        mean = lambda v: np.bincount(inverse, v, len(counts)) / counts
        points = np.stack([mean(points[:, k]) for k in range(3)], 1)
        intensity = mean(intensity) if intensity is not None else None
    return points, intensity


def scale(values):
    """0..1 over the 2nd..98th percentile."""
    lo, hi = np.percentile(values, [2, 98])
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def equalized(values):
    """0..1 by rank: each colour gets as many points (intensity bunches at the low end, and per lidar)."""
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    below = np.cumsum(counts) - counts
    return ((below + counts / 2) / len(values))[inverse.reshape(-1)]


def rainbow(t):
    """pcl_viewer's scale, low blue to high red: full saturation, no purple."""
    h = (1.0 - t) * 4.0   # 4 blue, 3 cyan, 2 green, 1 yellow, 0 red
    x = 1.0 - np.abs(h % 2.0 - 1.0)
    one, zero = np.ones_like(t), np.zeros_like(t)
    sector = np.minimum(h.astype(int), 3)
    r = np.choose(sector, [one, x, zero, zero])
    g = np.choose(sector, [x, one, one, x])
    b = np.choose(sector, [zero, zero, x, one])
    return np.stack([r, g, b], 1)


def grey(t):
    return np.repeat((0.25 + 0.75 * t)[:, None], 3, 1)


SCALES = {"Rainbow": rainbow, "Turbo": lambda t: turbo(0.08 + 0.87 * t), "Grey": grey}


def turbo(t):
    t = np.clip(t, 0.0, 1.0)
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.22930659 + t * (-89.69712180 + t * 27.34824973))))
    return np.clip(np.stack([r, g, b], 1), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="a recording folder (or .mcap)")
    ap.add_argument("--pcd", default="", help="the map, in the odometry's frame")
    ap.add_argument("--odom", default="/helios_lio/odom", help="the LIO's odometry (/lio/odom for Point-LIO)")
    ap.add_argument("--antenna", default="0.221,0,0.0888", help="the GPS antenna on the base, x,y,z m")
    ap.add_argument("--fit_share", type=float, default=0.15,
                    help="the GPS is aligned on this first share of the RTK fixed fixes (1: all of them)")
    ap.add_argument("--sigma", type=float, default=1.0, help="circles of this many times the reported accuracy")
    ap.add_argument("--voxel", type=float, default=0.05, help="m, the map's downsampling")
    ap.add_argument("--color", choices=["height", "intensity"], default="height",
                    help="the map's colour (intensity: each lidar's own scale, see the note above)")
    ap.add_argument("--png", default="", help="also save a top-down image here")
    ap.add_argument("--no-3d", action="store_true", help="no 3D window (with --png)")
    ap.add_argument("--screenshot", default="", help="open the window, save its view here and close it")
    args = ap.parse_args()

    odom, fixes, status = read_bag(args.bag, args.odom)
    antenna = np.array([float(v) for v in args.antenna.split(",")])
    rows = [(f, solution(f, status)) for f in fixes]
    rows = [(f, s) for f, s in rows if s and odom[0, 0] <= f[0] <= odom[-1, 0]]
    if len(rows) < 3:
        sys.exit("fewer than 3 GPS fixes inside the odometry's time span")
    t = np.array([f[0] for f, _ in rows])
    kind = np.array([s for _, s in rows])
    h_std = np.array([f[6] for f, _ in rows])
    geo = np.array([f[3:6] for f, _ in rows])
    g_enu = enu(geo[:, 0], geo[:, 1], geo[:, 2], geo[0])
    b_lio, rot = base_poses(odom, t)
    a_lio = b_lio + rot.apply(antenna)   # the antenna, where the LIO says it was

    # Fit on the first share of the RTK fixed fixes (at least 10), else of the RTK ones, else of every fix
    for kinds in (["RTK fixed"], ["RTK fixed", "RTK float"], list(STYLE)):
        candidates = np.flatnonzero(np.isin(kind, kinds))
        if len(candidates) >= 10:
            break
    use = np.zeros(len(rows), bool)
    use[candidates[:max(10, int(np.ceil(args.fit_share * len(candidates))))]] = True
    fit_span = t[use][-1] - t[use][0]
    fit_path = np.sum(np.linalg.norm(np.diff(a_lio[use, :2], axis=0), axis=1))
    to_map, yaw = align(g_enu[use], a_lio[use], 1.0 / np.maximum(h_std[use], 0.01) ** 2)
    g_map = to_map(g_enu)
    err = np.linalg.norm(g_map[:, :2] - a_lio[:, :2], axis=1)
    # The base from the GPS: each fix moved back from the antenna, with the LIO's attitude at that time (the GPS
    # has none). Its distance to the LIO's base is the same as the antennas'.
    g_base = g_map - rot.apply(antenna)

    name = os.path.basename(os.path.normpath(args.bag))
    path_len = np.sum(np.linalg.norm(np.diff(odom[:, 1:3], axis=0), axis=1))
    print(f"\n  {name}: {path_len:.0f} m of LIO path, {len(rows)} GPS fixes; "
          f"GPS turned {yaw:.1f} deg onto the LIO frame (fit on the first {use.sum()} {' / '.join(kinds)} fixes: "
          f"{fit_span:.0f} s, {fit_path:.0f} m of path)")
    print(f"  {'solution':10s} {'fixes':>6s} {'share':>6s} {'reported (median)':>18s} {'GPS - LIO xy mean':>18s} "
          f"{'median':>7s} {'within 1/2 sigma':>17s}")
    summary = [f"{path_len:.0f} m path, {len(rows)} fixes", f"aligned on the first {use.sum()} {kinds[-1]}",
               f"  ({fit_span:.0f} s, {fit_path:.0f} m)", "GPS - SLAM, xy (mean / median):"]
    for label in STYLE:
        m = kind == label
        if m.any():
            within = [np.mean(err[m] <= k * np.maximum(h_std[m], 1e-3)) * 100 for k in (1, 2)]
            print(f"  {label:10s} {m.sum():6d} {m.mean() * 100:5.0f}% {np.median(h_std[m]):17.3f}m "
                  f"{np.mean(err[m]):17.3f}m {np.median(err[m]):6.3f}m {within[0]:8.0f}% {within[1]:5.0f}%")
            summary.append(f"  {label}: {m.sum()}, {np.mean(err[m]) * 100:.1f} / {np.median(err[m]) * 100:.1f} cm")
    print("  (GPS - LIO includes the LIO's drift: over a long path it, not the GPS, can dominate)\n")

    cloud = None
    if args.pcd:
        points, intensity = load_map(os.path.expanduser(args.pcd), args.voxel)
        colors = {"Height": scale(points[:, 2])}
        if intensity is not None:
            colors["Intensity"] = equalized(intensity)
        elif args.color == "intensity":
            print("  the map has no intensity: coloured by height")
        first = "Intensity" if args.color == "intensity" and "Intensity" in colors else "Height"
        cloud = {"points": points, "colors": colors, "first": first}
        print(f"  map: {len(points):,} points after {args.voxel} m voxels")

    if args.png:
        png_cloud = (cloud["points"], rainbow(cloud["colors"][cloud["first"]])) if cloud else None
        save_png(args.png, png_cloud, odom, g_base, kind, h_std, args.sigma, t, err)
    if not args.no_3d:
        Viewer(cloud, odom, g_base, kind, h_std, args.sigma, name, summary, args.screenshot).run()


def save_png(path, cloud, odom, g_base, kind, h_std, sigma, t, err):
    """Left: top-down, around the path. Right: GPS - LIO distance over time against the reported accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    bg, fg = tuple(BG_COLOR), "white"
    fig, (ax, at) = plt.subplots(1, 2, figsize=(20, 10), facecolor=bg, gridspec_kw={"width_ratios": [1, 1]})
    for a in (ax, at):
        a.set_facecolor(bg)
        a.tick_params(colors=fg)
        for side in a.spines.values():
            side.set_color(fg)
    lo, hi = np.vstack([odom[:, 1:3], g_base[:, :2]]).min(0) - 8, np.vstack([odom[:, 1:3], g_base[:, :2]]).max(0) + 8
    if cloud is not None:
        p, c = cloud
        inside = (p[:, 0] > lo[0]) & (p[:, 0] < hi[0]) & (p[:, 1] > lo[1]) & (p[:, 1] < hi[1])
        p, c = p[inside], c[inside]
        pick = np.random.default_rng(0).choice(len(p), min(len(p), 600_000), replace=False)
        ax.scatter(p[pick, 0], p[pick, 1], s=0.15, c=c[pick] * 0.5, linewidths=0)
    ax.plot(odom[:, 1], odom[:, 2], color=fg, lw=1.0, label="base from the SLAM (LIO)")
    ax.plot(g_base[:, 0], g_base[:, 1], color=GPS_LINE, lw=0.8, label="base from the GPS")
    for name, rgb in STYLE.items():
        m = kind == name
        if not m.any():
            continue
        ax.scatter(g_base[m, 0], g_base[m, 1], s=6, color=rgb, label=f"GPS {name} ({m.sum()})", zorder=3)
        for (x, y), r in zip(g_base[m, :2], h_std[m]):
            ax.add_patch(Circle((x, y), sigma * r, fill=False, color=rgb, lw=0.5, alpha=0.6))
        at.scatter(t[m] - t[0], err[m] * 100, s=6, color=rgb, label=f"GPS {name}: GPS - LIO")
    at.plot(t - t[0], sigma * h_std * 100, color=fg, lw=0.8, alpha=0.8, label=f"reported accuracy ({sigma:g} sigma)")
    at.set_yscale("log")
    at.set_xlabel("s", color=fg)
    at.set_ylabel("horizontal, cm", color=fg)
    at.set_title("GPS - LIO distance and the receiver's reported accuracy", color=fg)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(f"The base from the SLAM and from the GPS ({sigma:g} sigma circles), m", color=fg)
    for a in (ax, at):
        leg = a.legend(loc="upper right", facecolor=bg, edgecolor=fg, fontsize=9)
        for text in leg.get_texts():
            text.set_color(fg)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=bg)
    print(f"  saved {path}")


class Viewer:
    """The 3D window: the scene, and a panel to colour the map, size and fade its points, size the GPS dots and
    error circles, and show or hide each layer. The camera turns about a centre with z kept up (a level horizon)."""

    def __init__(self, cloud, odom, g_base, kind, h_std, sigma, name, summary, screenshot=""):
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        self.o3d, self.gui, self.rendering = o3d, gui, rendering
        self.cloud, self.odom, self.g_base, self.kind, self.h_std = cloud, odom, g_base, kind, h_std
        self.sigma, self.screenshot = sigma, screenshot
        self.point_size, self.opacity, self.dot_size = 1.0, 1.0, 8.0
        self.map_on, self.circles_on, self.kind_on = True, True, {}
        # The map's height range above the path, for the cut slider (canopy hides the path from above)
        self.z_ground = float(np.median(odom[:, 3]))
        self.z_top = float(cloud["points"][:, 2].max()) - self.z_ground + 0.1 if cloud else 0.0
        self.cut = self.z_top
        self.color = cloud["first"] if cloud else "Height"
        self.scale = "Rainbow"
        self.kinds = [k for k in STYLE if (kind == k).any()]
        self.fov, self.drag = 45.0, None

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(f"SLAM and GPS - {name}", 1500, 950)
        em = self.window.theme.font_size
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.set_on_mouse(self.on_mouse)
        self.scene = self.widget.scene
        self.scene.set_background([*BG_COLOR, 1.0])
        self.scene.show_axes(False)
        # Flat light for the transparent (lit) map: no sun, so no shadows that move with the view, and the ambient
        # light at the brightness of the unlit shader (measured: 40000 with a matte material)
        self.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0.577, -0.577, -0.577])
        self.scene.scene.enable_sun_light(False)
        self.scene.scene.set_indirect_light_intensity(40000)

        self.panel = gui.ScrollableVert(0.4 * em, gui.Margins(em, em, em, em))
        self.build_panel(em, summary)
        self.window.add_child(self.widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self.on_layout)

        self.add_map()
        self.add_line("slam", odom[:, 1:4], (1.0, 1.0, 1.0), 3.0)
        self.add_line("gps_path", g_base, GPS_LINE, 2.0)
        self.add_gps()
        every = np.vstack([odom[:, 1:4], g_base] + ([cloud["points"]] if cloud else []))
        self.lo, self.hi = np.percentile(every, 1, axis=0), np.percentile(every, 99, axis=0)
        self.view("top")

    # ---- panel
    def build_panel(self, em, summary):
        gui = self.gui
        p = self.panel

        def section(title):
            p.add_fixed(0.5 * em)
            label = gui.Label(title)
            label.text_color = gui.Color(0.6, 0.85, 1.0)
            p.add_child(label)

        def combo(title, items, value, on_change):
            row = gui.Horiz(0.5 * em)
            row.add_child(gui.Label(title))
            box = gui.Combobox()
            for item in items:
                box.add_item(item)
            box.selected_text = value
            box.set_on_selection_changed(on_change)
            row.add_child(box)
            p.add_child(row)

        def slider(title, kind, lo, hi, value, on_change):
            row = gui.Horiz(0.5 * em)
            row.add_child(gui.Label(title))
            sl = gui.Slider(kind)
            sl.set_limits(lo, hi)
            if kind == gui.Slider.INT:
                sl.int_value = int(value)
            else:
                sl.double_value = value
            sl.set_on_value_changed(on_change)
            row.add_child(sl)
            p.add_child(row)

        def checkbox(title, on_change):
            box = gui.Checkbox(title)
            box.checked = True
            box.set_on_checked(on_change)
            p.add_child(box)

        if self.cloud:
            section("Map")
            combo("Colour by", list(self.cloud["colors"]), self.color, self.on_color)
            combo("Scale   ", list(SCALES), self.scale, self.on_scale)
            slider("Point size", gui.Slider.DOUBLE, 0.2, 3.0, self.point_size, self.on_point_size)
            slider("Opacity   ", gui.Slider.DOUBLE, 0.05, 1.0, self.opacity, self.on_opacity)
            slider("Cut above, m", gui.Slider.DOUBLE, 0.0, max(self.z_top, 0.5), self.cut, self.on_cut)
        section("GPS")
        slider("Dot size  ", gui.Slider.INT, 2, 20, self.dot_size, self.on_dot_size)
        slider("Circles, sigma", gui.Slider.DOUBLE, 0.0, 3.0, self.sigma, self.on_sigma)
        section("Show")
        if self.cloud:
            checkbox("Map", self.show_map)
        checkbox("Base from the SLAM (white)", lambda on: self.scene.show_geometry("slam", on))
        checkbox("Base from the GPS (magenta)", lambda on: self.scene.show_geometry("gps_path", on))
        for k in self.kinds:
            checkbox(f"GPS {k} ({(self.kind == k).sum()})", lambda on, k=k: self.show_kind(k, on))
        checkbox("Error circles", self.on_circles)
        section("View")
        row = gui.Horiz(0.5 * em)
        for title, which in (("Top", "top"), ("Side", "side")):
            b = gui.Button(title)
            b.set_on_clicked(lambda which=which: self.view(which))
            row.add_child(b)
        p.add_child(row)
        section("Statistics")
        for line in summary:
            p.add_child(gui.Label(line))
        p.add_fixed(0.5 * em)
        p.add_child(gui.Label("green RTK fixed, amber float,\norange DGNSS, red GNSS"))
        p.add_child(gui.Label("Cut above: hides the map over that\nheight above the path (tree tops)"))
        p.add_child(gui.Label("Point size under 1: 1 pixel, faded\nby the pixel it would cover"))
        p.add_child(gui.Label("Mouse: left turns, middle or\nshift+left pans, right or wheel zooms"))

    def on_layout(self, _):
        r = self.window.content_rect
        width = 22 * self.window.theme.font_size
        self.widget.frame = self.gui.Rect(r.x, r.y, r.width - width, r.height)
        self.panel.frame = self.gui.Rect(r.get_right() - width, r.y, width, r.height)
        if hasattr(self, "center"):
            self.apply_camera()

    # ---- geometry
    def material(self, shader, rgb=(1.0, 1.0, 1.0), alpha=1.0, point_size=None, line_width=None):
        m = self.rendering.MaterialRecord()
        m.shader = shader
        m.base_color = np.array([*rgb, alpha])
        if point_size is not None:
            m.point_size = point_size
        if line_width is not None:
            m.line_width = line_width
        return m

    def map_material(self):
        # GPUs draw no point under 1 pixel: smaller ones are 1 pixel, faded by the share of it they would cover.
        # The transparent shader is the lit one: Open3D 0.19's unlit one aborts on point clouds (Filament: uniform
        # "srgbColor" not found)
        alpha = self.opacity * min(self.point_size, 1.0) ** 2
        if alpha >= 0.999:
            return self.material("defaultUnlit", point_size=self.point_size)
        m = self.material("defaultLitTransparency", alpha=alpha, point_size=max(self.point_size, 1.0))
        m.base_roughness, m.base_reflectance, m.base_metallic = 1.0, 0.0, 0.0   # matte: no highlights
        return m

    def add_map(self):
        if not self.cloud:
            return
        keep = self.cloud["points"][:, 2] - self.z_ground <= self.cut
        pc = self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(self.cloud["points"][keep]))
        pc.colors = self.o3d.utility.Vector3dVector(SCALES[self.scale](self.cloud["colors"][self.color][keep]))
        # Re-added for every change: modify_geometry_material can crash Filament ("srgbColor" not found)
        if self.scene.has_geometry("map"):
            self.scene.remove_geometry("map")
        self.scene.add_geometry("map", pc, self.map_material())
        self.scene.scene.geometry_shadows("map", False, False)
        self.scene.show_geometry("map", self.map_on)

    def add_line(self, name, points, rgb, width):
        ls = self.o3d.geometry.LineSet()
        ls.points = self.o3d.utility.Vector3dVector(points)
        ls.lines = self.o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points) - 1)])
        self.scene.add_geometry(name, ls, self.material("unlitLine", rgb, line_width=width))

    def add_gps(self):
        for k in self.kinds:
            self.add_dots(k)
            self.add_circles(k)

    def add_dots(self, k):
        name = f"dots {k}"
        if self.scene.has_geometry(name):
            self.scene.remove_geometry(name)
        dots = self.o3d.geometry.PointCloud(self.o3d.utility.Vector3dVector(self.g_base[self.kind == k]))
        dots.paint_uniform_color(STYLE[k])
        self.scene.add_geometry(name, dots, self.material("defaultUnlit", point_size=self.dot_size))
        self.scene.show_geometry(name, self.kind_on.get(k, True))

    def add_circles(self, k, ring=None):
        ring = np.linspace(0, 2 * np.pi, 33) if ring is None else ring
        name = f"circles {k}"
        if self.scene.has_geometry(name):
            self.scene.remove_geometry(name)
        m = self.kind == k
        if self.sigma <= 0:
            return
        pts, lines = [], []
        for c, r in zip(self.g_base[m], self.h_std[m]):
            base = len(pts)
            pts.extend(c + self.sigma * r * np.stack([np.cos(ring), np.sin(ring), np.zeros_like(ring)], 1))
            lines.extend([[base + i, base + i + 1] for i in range(len(ring) - 1)])
        circles = self.o3d.geometry.LineSet(self.o3d.utility.Vector3dVector(np.array(pts)),
                                            self.o3d.utility.Vector2iVector(lines))
        self.scene.add_geometry(name, circles, self.material("unlitLine", STYLE[k], line_width=1.5))
        self.scene.show_geometry(name, self.circles_on and self.kind_on.get(k, True))

    # ---- callbacks
    def on_color(self, text, _):
        self.color = text
        self.add_map()

    def on_scale(self, text, _):
        self.scale = text
        self.add_map()

    def on_point_size(self, value):
        self.point_size = float(value)
        self.add_map()

    def on_opacity(self, value):
        self.opacity = float(value)
        self.add_map()

    def show_map(self, on):
        self.map_on = on
        self.scene.show_geometry("map", on)

    def on_cut(self, value):
        self.cut = float(value)
        self.add_map()

    def on_dot_size(self, value):
        self.dot_size = float(value)
        for k in self.kinds:
            self.add_dots(k)

    def on_sigma(self, value):
        self.sigma = float(value)
        for k in self.kinds:
            self.add_circles(k)

    def show_kind(self, k, on):
        self.kind_on[k] = on
        self.scene.show_geometry(f"dots {k}", on)
        if self.scene.has_geometry(f"circles {k}"):
            self.scene.show_geometry(f"circles {k}", on and self.circles_on)

    def on_circles(self, on):
        self.circles_on = on
        for k in self.kinds:
            if self.scene.has_geometry(f"circles {k}"):
                self.scene.show_geometry(f"circles {k}", on and self.kind_on.get(k, True))

    # ---- camera: a turntable about self.center, z up
    def view(self, which):
        self.center = (self.lo + self.hi) / 2
        self.reach = float(np.max(self.hi - self.lo))
        self.distance = 1.2 * self.reach
        self.yaw = np.radians(-90.0)   # from the south, north up the screen
        self.pitch = np.radians(89.0 if which == "top" else 25.0)
        self.apply_camera()

    def axes(self):
        """The eye's direction from the centre, and the screen's right and up in the world."""
        cp, sp, cy, sy = np.cos(self.pitch), np.sin(self.pitch), np.cos(self.yaw), np.sin(self.yaw)
        back = np.array([cp * cy, cp * sy, sp])
        right = np.array([-sy, cy, 0.0])
        return back, right, np.cross(back, right)

    def apply_camera(self):
        back, _, up = self.axes()
        frame = self.widget.frame
        aspect = frame.width / max(frame.height, 1)
        self.scene.camera.set_projection(self.fov, aspect, max(self.distance * 1e-3, 0.01),
                                         self.distance + 3.0 * self.reach, self.rendering.Camera.FovType.Vertical)
        self.scene.camera.look_at(self.center, self.center + self.distance * back, up)
        self.widget.force_redraw()

    def on_mouse(self, e):
        gui = self.gui
        consumed = gui.Widget.EventCallbackResult.CONSUMED
        if e.type == gui.MouseEvent.Type.WHEEL:
            self.distance *= 1.15 ** e.wheel_dy
            self.apply_camera()
            return consumed
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
            if e.is_button_down(gui.MouseButton.LEFT):
                pan = e.is_modifier_down(gui.KeyModifier.SHIFT) or e.is_modifier_down(gui.KeyModifier.CTRL)
                self.drag = "pan" if pan else "turn"
            elif e.is_button_down(gui.MouseButton.MIDDLE):
                self.drag = "pan"
            elif e.is_button_down(gui.MouseButton.RIGHT):
                self.drag = "zoom"
            self.last = (e.x, e.y)
            return consumed
        if e.type == gui.MouseEvent.Type.DRAG and self.drag:
            dx, dy = e.x - self.last[0], e.y - self.last[1]
            self.last = (e.x, e.y)
            if self.drag == "turn":
                self.yaw -= np.radians(0.3 * dx)
                self.pitch = np.clip(self.pitch + np.radians(0.3 * dy), np.radians(-89.0), np.radians(89.0))
            elif self.drag == "pan":
                _, right, up = self.axes()
                per_pixel = 2.0 * self.distance * np.tan(np.radians(self.fov / 2)) / max(self.widget.frame.height, 1)
                self.center = self.center - right * dx * per_pixel + up * dy * per_pixel
            else:
                self.distance *= 1.01 ** dy
            self.apply_camera()
            return consumed
        if e.type == gui.MouseEvent.Type.BUTTON_UP:
            self.drag = None
            return consumed
        return gui.Widget.EventCallbackResult.IGNORED

    def run(self):
        if self.screenshot:
            import threading

            def grab():
                def save(image):
                    self.o3d.io.write_image(self.screenshot, image)
                    print(f"  saved {self.screenshot}")
                    self.app.quit()
                self.scene.scene.render_to_image(save)
            threading.Timer(4.0, lambda: self.app.post_to_main_thread(self.window, grab)).start()
        print("  The 3D window: the panel on the right colours, sizes and hides; left-drag turns (horizon level), "
              "middle-drag or shift+left pans, right-drag or scroll zooms")
        self.app.run()


if __name__ == "__main__":
    main()
