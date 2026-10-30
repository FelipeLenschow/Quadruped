"""Open-vocabulary object detection on one camera with YOLOE (config.yaml's detector:).

Reads /camera/<camera>/image_raw/compressed and publishes, at most at rate:
  /camera/<camera>/detections: JSON {"stamp", "frame", "width", "height", "ms",
                                     "objects": [{"label", "score", "box": [x1, y1, x2, y2],
                                                  "mask": [[x, y], ...]}]} (pixels; mask: the outline)
  /camera/<camera>/detections/image/compressed: the image with the boxes drawn (only while subscribed)

--export builds a TensorRT engine for the configured model, classes and imgsz. Engines are tied to the GPU
they were built on, so run it on the robot itself; afterwards the node loads the engine instead of the .pt.
"""

import argparse
import array
import contextlib
import hashlib
import json
import os
import signal
import threading
import time
import zlib

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from quadruped_common import paths
from quadruped_common.config_loader import load_config

JPEG = "bgr8; jpeg compressed bgr8"
DRAW_WIDTH = 640


def settings(cfg):
    model = paths.repo(cfg.get("model", "models/yoloe-26s-seg-pf.pt"))
    classes = [] if model.endswith("-pf.pt") else [str(c) for c in cfg.get("classes") or []]
    return model, classes, int(cfg.get("imgsz", 640))


def engine_path(model, classes, imgsz):
    tag = hashlib.sha1("\n".join(classes).encode()).hexdigest()[:8] if classes else "all"
    return f"{os.path.splitext(model)[0]}-{tag}-{imgsz}.engine"


def outline(points, tolerance_px=2.0):
    """A mask's outline with fewer points, as [[x, y], ...] in whole pixels; [] when there is none."""
    if points is None or len(points) < 3:
        return []
    poly = cv2.approxPolyDP(np.asarray(points, np.float32).reshape(-1, 1, 2), tolerance_px, True)
    return poly.reshape(-1, 2).round().astype(int).tolist()


def overlap(a, b):
    """The share of the smaller of boxes a and b ([x1, y1, x2, y2]) that lies inside the other."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-6)


def drop_overlaps(objects, limit):
    """objects without those mostly (limit) inside, or around, a higher-scoring one of the same label. YOLO26 has
    no NMS: a chair got a second box (0.50 and 0.39), and the mapper put the two apart as two chairs."""
    kept = []
    for o in sorted(objects, key=lambda o: -o["score"]):
        if all(k["label"] != o["label"] or overlap(k["box"], o["box"]) < limit for k in kept):
            kept.append(o)
    return kept


def label_color(label):
    return tuple(int(c) for c in np.random.default_rng(zlib.crc32(label.encode())).integers(64, 256, 3))


@contextlib.contextmanager
def inside(folder):
    """YOLOE downloads its text encoder into the working directory."""
    old = os.getcwd()
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(old)


def load_yoloe(model, classes):
    from ultralytics import YOLOE
    with inside(os.path.dirname(model)):
        m = YOLOE(os.path.basename(model))
        if classes:
            m.set_classes(classes)
    return m


def load(cfg):
    model, classes, imgsz = settings(cfg)
    engine = engine_path(model, classes, imgsz)
    if os.path.exists(engine):
        from ultralytics import YOLO
        return YOLO(engine, task="segment"), engine
    if not model.endswith(".pt"):
        raise FileNotFoundError(f"{model} is not a .pt and no engine at {engine}")
    if not classes and not model.endswith("-pf.pt"):
        raise ValueError(f"{os.path.basename(model)} needs classes: (or use a *-seg-pf.pt model)")
    return load_yoloe(model, classes), model


def export(cfg):
    model, classes, imgsz = settings(cfg)
    m = load_yoloe(model, classes)
    with inside(os.path.dirname(model)):
        out = m.export(format="engine", quantize=16, imgsz=imgsz, device=0)
        os.replace(out, engine_path(model, classes, imgsz))
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.splitext(out)[0] + ".onnx")
    print(f"[object_detector] engine: {engine_path(model, classes, imgsz)}")


class ObjectDetector(Node):
    def __init__(self, args, cfg):
        super().__init__("object_detector")
        self.camera = args.camera or cfg.get("camera", "front")
        self.period = 1.0 / float(args.rate or cfg.get("rate", 5))
        self.conf = float(cfg.get("conf", 0.25))
        self.class_conf = {str(k): float(v) for k, v in (cfg.get("class_conf") or {}).items()}
        self.max_overlap = float(cfg.get("max_overlap", 0.6))
        self.imgsz = int(cfg.get("imgsz", 640))
        # Load before touching CUDA: with torch.cuda.is_available() called first, a prompted model's set_classes
        # gave embeddings that matched nothing (0 objects on the Jetson; the same frame had a chair at 0.88).
        self.model, source = load(cfg)
        import torch
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.warm = False
        self.latest = None
        self.new = threading.Event()
        self.stop = threading.Event()
        base = f"/camera/{self.camera}"
        self.pub = self.create_publisher(String, f"{base}/detections", 2)
        self.image_pub = self.create_publisher(CompressedImage, f"{base}/detections/image/compressed", 1)
        self.create_subscription(CompressedImage, f"{base}/image_raw/compressed", self._image_cb, 1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.get_logger().info(f"{os.path.basename(source)} ({len(self.model.names)} names) on "
                               f"{'GPU' if self.device == 0 else 'CPU'}: {base}/image_raw/compressed -> "
                               f"{base}/detections, {1 / self.period:g} Hz")

    def _image_cb(self, m):
        self.latest = m
        self.new.set()

    def _run(self):
        while not self.stop.is_set():
            if not self.new.wait(0.5):
                continue
            self.new.clear()
            m, t0 = self.latest, time.monotonic()
            try:
                self._detect(m)
            except Exception as e:
                self.get_logger().error(f"detection failed: {e}", throttle_duration_sec=10.0)
            self.stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def _predict(self, image):
        conf = min([self.conf, *self.class_conf.values()])
        return self.model.predict(image, imgsz=self.imgsz, conf=conf, device=self.device,
                                  quantize=16 if self.device == 0 else None, verbose=False)[0]

    def _detect(self, m):
        image = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        if not self.warm:
            self._predict(image)
            self.warm = True
        t0 = time.monotonic()
        r = self._predict(image)
        ms = (time.monotonic() - t0) * 1000
        outlines = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        objects = [{"label": r.names[int(c)], "score": round(s, 3), "box": [round(v, 1) for v in b],
                    "mask": outline(m)}
                   for b, s, c, m in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(),
                                         outlines)
                   if s >= self.class_conf.get(r.names[int(c)], self.conf)]
        objects = drop_overlaps(objects, self.max_overlap)
        h, w = image.shape[:2]
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.pub.publish(String(data=json.dumps({"stamp": stamp, "frame": m.header.frame_id, "width": w,
                                                 "height": h, "ms": round(ms, 1), "objects": objects})))
        self.get_logger().info(f"{len(objects)} objects, {ms:.0f} ms: "
                               f"{', '.join(sorted({o['label'] for o in objects})) or '-'}",
                               throttle_duration_sec=10.0)
        if self.image_pub.get_subscription_count():
            self._publish_image(image, objects, m.header)

    def _publish_image(self, image, objects, header):
        s = min(1.0, DRAW_WIDTH / image.shape[1])
        if s < 1.0:
            image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for o in objects:
            x1, y1, x2, y2 = (int(v * s) for v in o["box"])
            color = label_color(o["label"])
            text = f"{o['label']} {o['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y = max(th + 4, y1)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image, (x1, y - th - 4), (x1 + tw + 4, y), color, -1)
            cv2.putText(image, text, (x1 + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        msg = CompressedImage(format=JPEG)
        msg.header = header
        msg.data = array.array("B", cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        self.image_pub.publish(msg)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="", help="default detector.camera")
    ap.add_argument("--rate", type=float, default=0.0, help="Hz, default detector.rate")
    ap.add_argument("--classes", default="", help="comma-separated, overrides detector.classes (a -pf model "
                                                  "is swapped for its -seg.pt)")
    ap.add_argument("--export", action="store_true", help="build the TensorRT engine and exit")
    args = ap.parse_args(remove_ros_args()[1:])

    cfg = load_config().get("detector") or {}
    if args.classes:
        model = cfg.get("model", "models/yoloe-26s-seg-pf.pt").replace("-seg-pf.pt", "-seg.pt")
        cfg = dict(cfg, model=model, classes=[c.strip() for c in args.classes.split(",") if c.strip()])
    if args.export:
        export(cfg)
        return

    rclpy.init()
    node = ObjectDetector(args, cfg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
