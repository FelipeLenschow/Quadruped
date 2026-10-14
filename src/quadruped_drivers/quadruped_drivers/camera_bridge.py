"""Cameras to ROS 2: every entry under config.yaml's cameras:, each from its own source (see there).

Per camera <name>: /camera/<name>/image_raw/compressed, /camera/<name>/image_raw (decoded, only while subscribed),
/camera/<name>/camera_info (once calibrated), /camera/<name>/preview/compressed (small, slow, only while
subscribed) and the mount TF. /cameras lists them with their preview settings (JSON, latched). Nothing is read
or encoded while nobody subscribes. camera_calibration's Commit (/camera/<name>/set_camera_info) writes the
calibration yaml. /camera/<name>/preview/settings ({"rate": Hz, "width": px}, JSON) changes the preview until
the bridge restarts. Cameras with source: unitree come from quadruped_real's go2_camera (the Go2's videohub),
subscribed to only while one of the camera's own topics is.
"""

import argparse
import array
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from sensor_msgs.srv import SetCameraInfo
from std_msgs.msg import String

from quadruped_common import paths
from quadruped_common.config_loader import load_config
from quadruped_drivers.sensor_mounts import optical_frame, publish_static_tf

JPEG = "bgr8; jpeg compressed bgr8"
PNG16 = "mono16; png compressed mono16"
STALE_S = 2.0
PREVIEW_RATE = (0.2, 15.0)
PREVIEW_WIDTH = (80, 1280)
GO2_CAMERA_TOPIC = "/go2_camera/image/compressed"  # go2_camera.cpp's kTopic
RAW_ENCODINGS = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "mono8": (np.uint8, 1), "mono16": (np.uint16, 1),
                 "16UC1": (np.uint16, 1), "32FC1": (np.float32, 1)}


def _bytes(data):
    out = array.array("B")
    out.frombytes(data)
    return out


def load_camera_info(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        c = yaml.safe_load(f)
    info = CameraInfo()
    info.width, info.height = c["image_width"], c["image_height"]
    info.distortion_model = c.get("distortion_model", "plumb_bob")
    info.k = [float(v) for v in c["camera_matrix"]["data"]]
    info.d = [float(v) for v in c["distortion_coefficients"]["data"]]
    info.r = [float(v) for v in c["rectification_matrix"]["data"]]
    info.p = [float(v) for v in c["projection_matrix"]["data"]]
    return info


def save_camera_info(path, info, name):
    """camera_calibration's own yaml layout, so its files and ours are interchangeable."""
    def mat(rows, cols, data):
        return {"rows": rows, "cols": cols, "data": [float(v) for v in data]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"image_width": info.width, "image_height": info.height, "camera_name": name,
                        "camera_matrix": mat(3, 3, info.k), "distortion_model": info.distortion_model,
                        "distortion_coefficients": mat(1, len(info.d), info.d),
                        "rectification_matrix": mat(3, 3, info.r), "projection_matrix": mat(3, 4, info.p)},
                       f, default_flow_style=None, sort_keys=False)


class V4l2Source:
    """Any camera OpenCV opens: USB/UVC webcams, and thermal cameras that give 16-bit frames (fourcc Y16)."""

    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(cfg.get("device", "/dev/video0"), cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {cfg.get('device', '/dev/video0')}")
        if cfg.get("fourcc"):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(cfg["fourcc"]).ljust(4)))
        for prop, key in ((cv2.CAP_PROP_FRAME_WIDTH, "width"), (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
                          (cv2.CAP_PROP_FPS, "fps")):
            if cfg.get(key):
                self.cap.set(prop, float(cfg[key]))
        if cfg.get("kind") == "thermal":
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, image = self.cap.read()
        return {"image": image} if ok else None

    def idle(self):
        self.cap.grab()

    def close(self):
        self.cap.release()


SOURCES = {"v4l2": V4l2Source}


class Camera:
    def __init__(self, node, name, cfg, sim):
        self.node, self.name, self.cfg = node, name, cfg
        self.log = node.get_logger()
        self.kind = cfg.get("kind", "rgb")
        self.frame_id = optical_frame(cfg.get("frame", f"{name}_camera"))
        self.period = 1.0 / float(cfg.get("rate", 15))
        self.preview_period = 1.0 / float(cfg.get("preview_rate", 2))
        self.preview_width = int(cfg.get("preview_width", 320))
        self.width = None
        self.last_frame = self.next_preview = 0.0
        self.stale = False
        base = f"/camera/{name}"
        self.preview_topic = f"{base}/preview/compressed"
        self.image_pub = node.create_publisher(CompressedImage, f"{base}/image_raw/compressed", 2)
        self.raw_pub = node.create_publisher(Image, f"{base}/image_raw", 2)
        self.preview_pub = node.create_publisher(CompressedImage, self.preview_topic, 1)
        node.create_subscription(String, f"{base}/preview/settings", self._settings_cb, 2)
        # Gazebo publishes the simulated camera's own camera_info.
        self.info_pub = None
        if not sim:
            self.calibration = paths.repo(cfg.get("calibration") or os.path.join("calibration", f"{name}.yaml"))
            self.info = load_camera_info(self.calibration)
            self.info_pub = node.create_publisher(CameraInfo, f"{base}/camera_info", 2)
            node.create_service(SetCameraInfo, f"{base}/set_camera_info", self._set_info)
            if self.info is None:
                self.log.warn(f"{name}: no calibration at {self.calibration}, no camera_info until one is saved")

        self.source = None
        self.go2_sub = None
        self.stop = threading.Event()
        topic = cfg.get("sim_topic") if sim else cfg.get("topic")
        source = "ros" if sim else cfg.get("source", "unitree")
        if source == "ros":
            compressed = topic.endswith("/compressed")
            node.create_subscription(CompressedImage if compressed else Image, topic,
                                     self._compressed_cb if compressed else self._raw_cb, 2)
            self.thread = None
        elif source == "unitree":
            topic = GO2_CAMERA_TOPIC
            node.create_timer(0.5, self._follow_go2)
            self.thread = None
        else:
            self.source = SOURCES[source](cfg)
            self.thread = threading.Thread(target=self._run, daemon=True)
        self.log.info(f"{name}: {source}{' ' + topic if source in ('ros', 'unitree') else ''} -> {base}/..., "
                      f"{1 / self.period:.0f} Hz, preview {self.preview_width} px {1 / self.preview_period:.1f} Hz")

    def describe(self):
        return {"name": self.name, "kind": self.kind, "frame": self.frame_id, "preview": self.preview_topic,
                "rate": round(1 / self.period, 2), "preview_rate": round(1 / self.preview_period, 2),
                "preview_width": self.preview_width}

    def _settings_cb(self, msg):
        try:
            req = json.loads(msg.data)
            rate = min(max(float(req.get("rate", 1 / self.preview_period)), PREVIEW_RATE[0]), PREVIEW_RATE[1],
                       round(1 / self.period, 2))
            width = min(max(int(req.get("width", self.preview_width)), PREVIEW_WIDTH[0]), PREVIEW_WIDTH[1])
        except (ValueError, TypeError, AttributeError):
            self.log.warn(f"{self.name}: bad preview settings {msg.data!r}")
            return
        if (rate, width) == (round(1 / self.preview_period, 2), self.preview_width):
            return
        self.preview_period, self.preview_width = 1.0 / rate, width
        self.log.info(f"{self.name}: preview {width} px {rate:g} Hz")
        self.node.publish_list()

    def start(self):
        if self.thread:
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.source:
            self.source.close()

    def _watched(self):
        pubs = (self.image_pub, self.raw_pub, self.preview_pub, self.info_pub)
        return any(p is not None and p.get_subscription_count() for p in pubs)

    def _run(self):
        while not self.stop.is_set() and rclpy.ok():
            t0 = time.monotonic()
            if not self._watched():
                self.source.idle()
                self.stop.wait(0.1)
                continue
            try:
                frame = self.source.read()
            except Exception as e:
                self.log.error(f"{self.name}: {e}", throttle_duration_sec=10.0)
                frame = None
            if frame is None:
                self._check_stale(t0)
                self.stop.wait(0.1)
                continue
            self.publish(**frame)
            self.stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def _follow_go2(self):
        """Subscribed to go2_camera only while watched: it asks the robot for a frame only while subscribed."""
        watched = self._watched()
        if watched and self.go2_sub is None:
            self.go2_sub = self.node.create_subscription(CompressedImage, GO2_CAMERA_TOPIC, self._compressed_cb, 2)
        elif not watched and self.go2_sub is not None:
            self.node.destroy_subscription(self.go2_sub)
            self.go2_sub = None
        if watched:
            self._check_stale(time.monotonic())

    def _check_stale(self, now):
        if self.last_frame and not self.stale and now - self.last_frame > STALE_S:
            self.stale = True
            self.log.warn(f"{self.name}: no image for {STALE_S:.0f} s")

    def _raw_cb(self, m):
        dtype, ch = RAW_ENCODINGS.get(m.encoding, (None, 0))
        if dtype is None or not self._due():
            return
        row = np.frombuffer(bytes(m.data), dtype).reshape(m.height, -1)[:, :m.width * ch]
        image = row.reshape(m.height, m.width, ch) if ch > 1 else row
        if m.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.publish(image=image, stamp=m.header.stamp)

    def _compressed_cb(self, m):
        if self._due():
            self.publish(data=bytes(m.data), format=m.format, stamp=m.header.stamp)

    def _due(self):
        return self._watched() and time.monotonic() - self.last_frame >= self.period * 0.9

    def publish(self, data=None, format=None, image=None, stamp=None):
        now = time.monotonic()
        if not self.last_frame or self.stale:
            shape = image.shape if image is not None else f"{len(data) // 1024} kB, {format}"
            self.log.info(f"{self.name}: {'image resumed' if self.stale else 'first image'} ({shape})")
        self.last_frame, self.stale = now, False
        if stamp is None:
            stamp = self.node.get_clock().now().to_msg()

        if self.image_pub.get_subscription_count():
            if data is None:
                data, format = self._encode(image)
            msg = CompressedImage(format=format)
            msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
            msg.data = _bytes(data)
            self.image_pub.publish(msg)

        if self.raw_pub.get_subscription_count():
            image = self._decode(data, format, image)
            self._publish_raw(image, stamp)

        if self.info_pub is not None and self.info is not None and self.info_pub.get_subscription_count():
            self.info.header.stamp, self.info.header.frame_id = stamp, self.frame_id
            self.info_pub.publish(self.info)

        # A schedule, not a minimum gap: 10 Hz from a 15 Hz camera then alternates gaps of 1 and 2 frames.
        if self.preview_pub.get_subscription_count() and now >= self.next_preview:
            self.next_preview = max(self.next_preview + self.preview_period, now)
            self._publish_preview(self._decode(data, format, image, self.preview_width), stamp)

    def _decode(self, data, format, image, min_width=None):
        """The frame as an array; for a preview, a JPEG is decoded at 1/2, 1/4 or 1/8 size when that is still
        at least min_width wide (much faster than decoding it whole)."""
        if image is not None:
            return image
        flag = cv2.IMREAD_UNCHANGED
        if min_width and self.width and "jpeg" in format:
            for k, f in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                         (2, cv2.IMREAD_REDUCED_COLOR_2)):
                if self.width // k >= min_width:
                    flag = f
                    break
        image = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
        if flag == cv2.IMREAD_UNCHANGED and image is not None:
            self.width = image.shape[1]
        return image

    def _encode(self, image):
        if image.dtype == np.uint16:
            return cv2.imencode(".png", image)[1].tobytes(), PNG16
        return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), JPEG

    def _publish_raw(self, image, stamp):
        if image is None:
            return
        ch = 1 if image.ndim == 2 else image.shape[2]
        encoding = {(np.dtype(np.uint8), 3): "bgr8", (np.dtype(np.uint8), 1): "mono8",
                    (np.dtype(np.uint16), 1): "mono16", (np.dtype(np.float32), 1): "32FC1"}.get((image.dtype, ch))
        if encoding is None:
            return
        msg = Image(height=image.shape[0], width=image.shape[1], encoding=encoding,
                    step=image.shape[1] * ch * image.dtype.itemsize)
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        msg.data = _bytes(np.ascontiguousarray(image).tobytes())
        self.raw_pub.publish(msg)

    def _publish_preview(self, image, stamp):
        if image is None:
            return
        if image.dtype != np.uint8 or (self.kind == "thermal" and image.ndim == 2):
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            image = cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)
        h, w = image.shape[:2]
        if w > self.preview_width:
            image = cv2.resize(image, (self.preview_width, round(h * self.preview_width / w)),
                               interpolation=cv2.INTER_AREA)
        msg = CompressedImage(format=JPEG)
        msg.header.stamp, msg.header.frame_id = stamp, self.frame_id
        msg.data = _bytes(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes())
        self.preview_pub.publish(msg)

    def _set_info(self, req, res):
        try:
            save_camera_info(self.calibration, req.camera_info, self.name)
        except OSError as e:
            res.success, res.status_message = False, str(e)
            return res
        self.info = req.camera_info
        res.success, res.status_message = True, self.calibration
        self.log.info(f"{self.name}: calibration saved to {self.calibration}")
        return res


class CameraBridge(Node):
    def __init__(self, args, cams):
        super().__init__("camera_bridge")
        self.cameras = []
        for name, cfg in cams.items():
            try:
                self.cameras.append(Camera(self, name, cfg, args.sim))
            except Exception as e:
                self.get_logger().error(f"{name}: not started: {e}")
        mounts = {c.cfg.get("frame", f"{c.name}_camera"): (tuple(float(v) for v in c.cfg.get("xyz", (0, 0, 0))),
                                                          tuple(float(v) for v in c.cfg.get("rpy", (0, 0, 0))))
                  for c in self.cameras}
        self.tf = publish_static_tf(self, args.base_frame, mounts, cameras=list(mounts))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.list_pub = self.create_publisher(String, "/cameras", latched)
        self.publish_list()
        for c in self.cameras:
            c.start()

    def publish_list(self):
        self.list_pub.publish(String(data=json.dumps([c.describe() for c in self.cameras])))

    def close(self):
        for c in self.cameras:
            c.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", action="store_true", help="Gazebo: read each camera's sim_topic")
    ap.add_argument("--only", default="", help="comma-separated camera names (default all)")
    ap.add_argument("--base_frame", default="base")
    args = ap.parse_args(remove_ros_args()[1:])

    cfg = load_config()
    cams = cfg.get("cameras") or {}
    if args.only:
        cams = {k: v for k, v in cams.items() if k in args.only.split(",")}
    if args.sim:
        cams = {k: v for k, v in cams.items() if v.get("sim_topic")}
    if not cams:
        print("[camera_bridge] no cameras configured")
        return

    if not args.sim:
        os.environ["ROS_DOMAIN_ID"] = str(cfg.get("network", {}).get("ros_domain_id", "1"))

    rclpy.init()
    node = CameraBridge(args, cams)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
