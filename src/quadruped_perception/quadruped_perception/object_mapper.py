"""Objects on the map: object_detector's sightings placed in the map frame and merged (config.yaml's object_mapper:).

For each detection, the lidar points of the last keep_s seconds (kept in the map frame, so the robot can move)
are projected onto the image; those inside the object's outline give its position, from the nearest cluster so
the wall behind it does not count. With too few points, the ray through the box's bottom-centre is crossed with
the floor. Both see the object's front, so the position is pushed back by half its width (from how far its
points spread sideways, or from the box), up to max_depth_shift. A sighting within radius of a known object
updates it, and labels are voted, so a few odd names ("extrude" up close) do not stick.

Objects of a forget_unseen label (people move) are removed after that many detector frames in which they should
be in clear view but are not detected: inside the image, within forget_range, and no lidar points nearer along the
way. Other labels are never removed, as the detector can miss an object from one angle.

Publishes /objects (JSON, latched): {"frame", "objects": [{"id", "label", "x", "y", "z", "sightings",
"last_seen", "labels": {label: votes}}]}, only objects seen min_sightings times, and /objects/markers for rviz.
Any message on /objects/clear forgets them all.
"""

import argparse
import json
import signal
from collections import Counter, deque

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from quadruped_common.config_loader import load_config
from quadruped_perception.camera_lidar_check import pixel_ray, project, xyz
from quadruped_perception.lidar_filter import rotation
from quadruped_perception.object_detector import label_color

CLUSTER_M = 0.3
OCCLUDER_M = 0.4


class ObjectMapper(Node):
    def __init__(self, camera, cfg):
        super().__init__("object_mapper")
        self.frame = cfg.get("frame", "map")
        self.keep_s = float(cfg.get("keep_s", 1.0))
        self.max_range = float(cfg.get("max_range", 8.0))
        self.radius = float(cfg.get("radius", 0.5))
        self.min_sightings = int(cfg.get("min_sightings", 3))
        self.min_points = int(cfg.get("min_points", 3))
        self.floor_z = float(cfg.get("floor_z", 0.0))
        self.floor_margin = float(cfg.get("floor_margin", 0.12))
        self.max_box = float(cfg.get("max_box_fraction", 0.5))
        self.max_width = float(cfg.get("max_box_width", 0.9))
        self.max_shift = float(cfg.get("max_depth_shift", 0.3))
        self.ignore = set(cfg.get("ignore") or [])
        self.forget = {str(k): int(v) for k, v in (cfg.get("forget_unseen") or {}).items()}
        self.forget_range = float(cfg.get("forget_range", 4.0))
        self.info = None
        self.clouds = deque()
        self.objects = []
        self.next_id = 1
        self.changed = True
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        base = f"/camera/{camera}"
        cloud = cfg.get("cloud", "/lidar/points_filtered")
        self.create_subscription(CameraInfo, f"{base}/camera_info", self._info_cb, 2)
        self.create_subscription(String, f"{base}/detections", self._detections_cb, 5)
        self.create_subscription(PointCloud2, cloud, self._cloud_cb, qos_profile_sensor_data)
        self.create_subscription(String, "/objects/clear", self._clear_cb, 2)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, "/objects", latched)
        self.marker_pub = self.create_publisher(MarkerArray, "/objects/markers", latched)
        self.create_timer(0.5, self._publish)
        self.get_logger().info(f"{base}/detections + {cloud} -> /objects in {self.frame}")

    def _info_cb(self, m):
        self.info = m

    def _clear_cb(self, _):
        self.objects, self.next_id, self.changed = [], 1, True
        self.get_logger().info("objects cleared")

    def _transform(self, target, source, stamp):
        """(R, t) taking points from source to target at stamp, or at the latest transform when that one is not
        buffered (yet)."""
        err = None
        for when in (stamp, Time()):
            try:
                t = self.tf_buffer.lookup_transform(target, source, when).transform
                return rotation(t.rotation), np.array([t.translation.x, t.translation.y, t.translation.z])
            except TransformException as e:
                err = e
        self.get_logger().warn(f"{source} -> {target}: {err}", throttle_duration_sec=5.0)
        return None

    def _cloud_cb(self, m):
        pts = xyz(m)
        tf = self._transform(self.frame, m.header.frame_id, Time.from_msg(m.header.stamp)) if len(pts) else None
        if tf is None:
            return
        rot, t = tf
        now = Time.from_msg(m.header.stamp).nanoseconds * 1e-9
        self.clouds.append((now, pts @ rot.T + t))
        while now - self.clouds[0][0] > self.keep_s:
            self.clouds.popleft()

    def _detections_cb(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        if self.info is None or not any(self.info.k):
            self.get_logger().warn("no camera_info: calibrate the camera first", throttle_duration_sec=5.0)
            return
        w, h = d["width"], d["height"]
        objects = [o for o in d["objects"] if o["label"] not in self.ignore and
                   o["box"][2] - o["box"][0] <= self.max_width * w and
                   (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) <= self.max_box * w * h]
        watched = [ob for ob in self.objects if label_of(ob) in self.forget]
        if not objects and not watched:
            return
        stamp = Time(nanoseconds=int(d["stamp"] * 1e9))
        to_cam = self._transform(d["frame"], self.frame, stamp)
        to_map = self._transform(self.frame, d["frame"], stamp)
        if to_cam is None or to_map is None:
            return
        scale = np.array([self.info.width / w, self.info.height / h])
        pts = np.vstack([c for _, c in self.clouds]) if self.clouds else np.zeros((0, 3))
        cam = pts @ to_cam[0].T + to_cam[1]
        dist = np.linalg.norm(cam, axis=1)
        # Not the floor: an outline's lower edge takes in the floor in front of the object, nearer than it.
        keep = (cam[:, 2] > 0.1) & (dist < self.max_range) & (pts[:, 2] > self.floor_z + self.floor_margin)
        pts, dist = pts[keep], dist[keep]
        uv = project(self.info, cam[keep]) / scale if keep.any() else np.zeros((0, 2))
        for o in objects:
            p, source = self._locate(o, uv, pts, dist, to_map, scale, h)
            if p is not None:
                self._add(o["label"], p, source, d["stamp"])
        if watched:
            self._forget_unseen(watched, d["objects"], to_cam, uv, dist, scale, w, h)

    def _forget_unseen(self, watched, seen, to_cam, uv, dist, scale, w, h):
        rot, t = to_cam
        mx, my = 0.1 * w, 0.1 * h
        for ob in watched:
            label = label_of(ob)
            c = rot @ ob["pos"] + t
            r = np.linalg.norm(c)
            if c[2] < 0.3 or r > self.forget_range:
                continue
            u, v = project(self.info, c[None])[0] / scale
            if not (mx <= u < w - mx and my <= v < h - my):
                continue
            if any(o["label"] == label and o["box"][0] - mx <= u <= o["box"][2] + mx and
                   o["box"][1] - my <= v <= o["box"][3] + my for o in seen):
                ob["misses"] = 0
                continue
            near = (np.abs(uv[:, 0] - u) < mx) & (np.abs(uv[:, 1] - v) < my)
            if np.count_nonzero(dist[near] < r - OCCLUDER_M) >= self.min_points:
                continue
            ob["misses"] = ob.get("misses", 0) + 1
        gone = [ob for ob in watched if ob.get("misses", 0) >= self.forget[label_of(ob)]]
        if gone:
            ids = {ob["id"] for ob in gone}
            self.objects = [ob for ob in self.objects if ob["id"] not in ids]
            self.changed = True
            self.get_logger().info("in view but not seen, removed: "
                                   + ", ".join(f"{label_of(ob)} {ob['id']}" for ob in gone))

    def _locate(self, o, uv, pts, dist, to_map, scale, height):
        """The object's position in the map frame and how it was found ("lidar" or "floor"), or (None, None)."""
        x1, y1, x2, y2 = o["box"]
        inside = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
        if o.get("mask") and inside.any():
            ox, oy = int(x1), int(y1)
            mask = np.zeros((int(y2) - oy + 1, int(x2) - ox + 1), np.uint8)
            cv2.fillPoly(mask, [np.array(o["mask"], np.int32) - (ox, oy)], 1)
            idx = np.flatnonzero(inside)
            u = np.clip(uv[idx, 0].astype(int) - ox, 0, mask.shape[1] - 1)
            v = np.clip(uv[idx, 1].astype(int) - oy, 0, mask.shape[0] - 1)
            inside[idx] = mask[v, u] > 0
        rot, origin = to_map
        near = dist[inside]
        if len(near) >= self.min_points:
            hits = pts[inside][near <= np.percentile(near, 20) + CLUSTER_M]
            p = np.median(hits, axis=0)
            side = self._behind(p, origin, 0.0)[1]
            spread = np.ptp(np.percentile(hits[:, :2] @ side, [5, 95]))
            return self._behind(p, origin, spread / 2)[0], "lidar"
        # The bottom of the box is where the object meets the floor, unless the image cuts it off.
        if y2 >= height - 2:
            return None, None
        ray = rot @ pixel_ray(self.info, (x1 + x2) / 2 * scale[0], y2 * scale[1])
        if ray[2] > -1e-3:
            return None, None
        p = origin + (self.floor_z - origin[2]) / ray[2] * ray
        r = np.linalg.norm(p - origin)
        if r >= self.max_range:
            return None, None
        width = (x2 - x1) * scale[0] / self.info.k[0] * r
        return self._behind(p, origin, width / 2)[0], "floor"

    def _behind(self, p, origin, shift):
        """p pushed away from the camera by shift (at most max_depth_shift) along the floor, and the floor's
        unit vector across the view there."""
        d = p[:2] - origin[:2]
        d = d / max(np.linalg.norm(d), 1e-6)
        out = np.array(p, float)
        out[:2] += d * min(shift, self.max_shift)
        return out, np.array([-d[1], d[0]])

    def _add(self, label, p, source, stamp):
        """Merges the sighting into the nearest object within radius (half of it when that object was never
        called label), or starts a new one."""
        best, best_d = None, None
        for ob in self.objects:
            d = np.hypot(*(ob["pos"][:2] - p[:2]))
            if d < (self.radius if label in ob["labels"] else self.radius / 2) and (best is None or d < best_d):
                best, best_d = ob, d
        if best is None:
            best = {"id": self.next_id, "pos": np.array(p, float), "n": 0, "labels": Counter(),
                    "sources": Counter()}
            self.next_id += 1
            self.objects.append(best)
        best["n"] += 1
        best["pos"] += max(1.0 / best["n"], 0.5 if label in self.forget else 0.05) * (p - best["pos"])
        best["labels"][label] += 1
        best["sources"][source] += 1
        best["last_seen"] = stamp
        best["misses"] = 0
        self.changed = True

    def _publish(self):
        if not self.changed:
            return
        self.changed = False
        shown = [ob for ob in self.objects if ob["n"] >= self.min_sightings]
        out = [{"id": ob["id"], "label": label_of(ob),
                "x": round(float(ob["pos"][0]), 3), "y": round(float(ob["pos"][1]), 3),
                "z": round(float(ob["pos"][2]), 3), "sightings": ob["n"], "last_seen": round(ob["last_seen"], 2),
                "labels": dict(ob["labels"].most_common(5))} for ob in shown]
        self.pub.publish(String(data=json.dumps({"frame": self.frame, "objects": out})))
        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for o in out:
            b, g, r = (c / 255 for c in label_color(o["label"]))
            for i, (kind, z, size) in enumerate(((Marker.CYLINDER, 0.15, (0.25, 0.25, 0.3)),
                                                 (Marker.TEXT_VIEW_FACING, 0.5, (0.0, 0.0, 0.2)))):
                mk = Marker(type=kind, action=Marker.ADD, ns="objects", id=o["id"] * 2 + i)
                mk.header.frame_id = self.frame
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = o["x"], o["y"], z
                mk.pose.orientation.w = 1.0
                mk.scale.x, mk.scale.y, mk.scale.z = size
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 0.9
                mk.text = f"{o['label']} ({o['sightings']})"
                markers.markers.append(mk)
        self.marker_pub.publish(markers)


def label_of(ob):
    return ob["labels"].most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="", help="default detector.camera")
    args = ap.parse_args(remove_ros_args()[1:])

    cfg = load_config()
    camera = args.camera or (cfg.get("detector") or {}).get("camera", "front")

    rclpy.init()
    node = ObjectMapper(camera, cfg.get("object_mapper") or {})
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
