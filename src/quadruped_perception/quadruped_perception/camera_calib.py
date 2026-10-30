"""Camera calibration without a window, for the robot (the image's OpenCV is headless, and there is no screen).

camera_calibration's own MonoCalibrator picks the samples and calibrates, as cameracalibrator does. Its
X / Y / size / skew progress is printed in the terminal and drawn on the image, which goes to the phone app as
an extra camera, "<name> calibration" (added to camera_bridge's /cameras while this runs). Move the board (or
the robot) until all four bars are full; it then calibrates and commits the result to camera_bridge through
/camera/<name>/set_camera_info, which writes calibration/<name>.yaml and starts publishing camera_info.
Keep still at each pose: moving frames are blurred and dropped.

After the commit it keeps running as a check: the image is undistorted, and with the board in view the
linear error (how far the board's corners are from straight lines, px) is printed and drawn. Under about
0.5 px everywhere, the edges too, is good; over 1 px, calibrate again (--rational for a wide lens).
"""

import argparse
import json
import signal
import time

import cv2
import rclpy
from camera_calibration.calibrator import ChessboardInfo, MonoCalibrator
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage, Image
from sensor_msgs.srv import SetCameraInfo
from std_msgs.msg import String

# OpenCV 5 returns the corners as (N, 2); camera_calibration 3.0.0 indexes them as OpenCV 4's (N, 1, 2).
_find_corners = cv2.findChessboardCorners


def _find_corners_n12(*args, **kwargs):
    ok, corners = _find_corners(*args, **kwargs)
    return ok, corners.reshape(-1, 1, 2) if corners is not None else corners


cv2.findChessboardCorners = _find_corners_n12


class CameraCalib(Node):
    def __init__(self, args):
        super().__init__("camera_calib")
        self.args = args
        board = ChessboardInfo()
        board.n_cols, board.n_rows, board.dim = args.cols, args.rows, args.square
        flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0
        self.cal = MonoCalibrator([board], flags)
        if args.fisheye:
            from camera_calibration.calibrator import CAMERA_MODEL
            self.cal.set_cammodel(CAMERA_MODEL.FISHEYE)
        base = f"/camera/{args.camera}"
        self.info_client = self.create_client(SetCameraInfo, f"{base}/set_camera_info")
        self.create_subscription(Image, f"{base}/image_raw", self._image_cb, qos_profile_sensor_data)
        self.last = None
        self.calibrating = False
        self.last_lin = 0.0

        # The app's camera list: camera_bridge's, with this node's image added while it runs.
        self.entry = {"name": f"{args.camera} calibration", "kind": "rgb", "frame": "",
                      "preview": f"{base}/calib/compressed", "rate": 15.0, "preview_rate": 2.0, "preview_width": 640}
        self.bridge_list = []
        self.next_preview = 0.0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.preview_pub = self.create_publisher(CompressedImage, self.entry["preview"], 1)
        self.list_pub = self.create_publisher(String, "/cameras", latched)
        self.create_subscription(String, "/cameras", self._list_cb, latched)
        self.create_subscription(String, f"{base}/calib/settings", self._settings_cb, 2)
        # Both nodes' latched lists reach a newly joined app in no set order: repeat ours so it wins.
        self.create_timer(1.0, self.publish_list)
        self.last_image = time.monotonic()
        self.create_timer(3.0, self._check_images)
        self.get_logger().info(f"{base}/image_raw, board {args.cols}x{args.rows} inner corners, "
                               f"{args.square * 1000:.1f} mm squares; in the app: camera '{self.entry['name']}'")

    def _list_cb(self, msg):
        try:
            cams = json.loads(msg.data)
        except ValueError:
            return
        if all(c.get("name") != self.entry["name"] for c in cams):
            self.bridge_list = cams
            self.publish_list()

    def publish_list(self, with_entry=True):
        self.list_pub.publish(String(data=json.dumps(self.bridge_list + ([self.entry] if with_entry else []))))

    def _settings_cb(self, msg):
        try:
            req = json.loads(msg.data)
            self.entry["preview_rate"] = min(max(float(req.get("rate", 2.0)), 0.2), 15.0)
            self.entry["preview_width"] = min(max(int(req.get("width", 640)), 160), 1280)
        except (ValueError, TypeError, AttributeError):
            return
        self.publish_list()

    def _check_images(self):
        if not self.calibrating and time.monotonic() - self.last_image > 3.0:
            self.get_logger().warn(f"no image on /camera/{self.args.camera}/image_raw: is camera_bridge ([I]) "
                                   "running?")

    def _image_cb(self, msg):
        self.last_image = time.monotonic()
        if self.calibrating:
            return
        drawn = self.cal.handle_msg(msg)
        if self.cal.calibrated:
            lin = drawn.linear_error
            text = f"check: linear error {lin:.2f} px" if lin >= 0 else "check: show the board"
            if lin >= 0 and time.monotonic() - self.last_lin > 1.0:
                print(text, flush=True)
                self.last_lin = time.monotonic()
            self._preview(drawn.scrib, [(text, None)])
            return
        params = drawn.params or []
        line = f"samples {len(self.cal.db):3d}  " + "  ".join(
            f"{name} {'#' * int(10 * min(1.0, p)):<10}" for name, _, _, p in params)
        if line != self.last:
            print(line, flush=True)
            self.last = line
        self._preview(drawn.scrib, [(f"samples {len(self.cal.db)}", None)]
                      + [(name, min(1.0, p)) for name, _, _, p in params])
        if self.cal.goodenough and len(self.cal.db) >= self.args.min_samples:
            self._calibrate()

    def _preview(self, img, rows):
        now = time.monotonic()
        if img is None or now < self.next_preview or self.count_subscribers(self.entry["preview"]) == 0:
            return
        self.next_preview = now + 1.0 / self.entry["preview_rate"]
        h, w = img.shape[:2]
        width = self.entry["preview_width"]
        img = cv2.resize(img, (width, int(h * width / w)))
        for i, (label, p) in enumerate(rows):
            y = 24 + 24 * i
            if p is not None:
                cv2.rectangle(img, (70, y - 14), (70 + 150, y), (255, 255, 255), 1)
                cv2.rectangle(img, (70, y - 14), (70 + int(150 * p), y), (0, 200, 0) if p >= 1 else (0, 200, 255), -1)
            cv2.putText(img, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.preview_pub.publish(CompressedImage(format="jpeg", data=jpg.tobytes()))

    def _calibrate(self):
        self.calibrating = True
        self.get_logger().info(f"calibrating from {len(self.cal.db)} samples (this takes a while)")
        self.cal.do_calibration()
        print(self.cal.ost(), flush=True)
        if not self.info_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"{self.info_client.srv_name} is not there: is camera_bridge running?")
            raise SystemExit(1)
        future = self.info_client.call_async(SetCameraInfo.Request(camera_info=self.cal.as_message()))
        future.add_done_callback(self._committed)

    def _committed(self, future):
        res = future.result()
        if not res.success:
            self.get_logger().error(f"commit failed: {res.status_message}")
            raise SystemExit(1)
        self.get_logger().info(f"committed: {res.status_message}. Now checking: show the board, the edges too; "
                               "Ctrl-C to quit")
        self.calibrating = False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--cols", type=int, default=8, help="inner corners across")
    ap.add_argument("--rows", type=int, default=5, help="inner corners down")
    ap.add_argument("--square", type=float, default=0.0439, help="square side, m")
    ap.add_argument("--min_samples", type=int, default=40, help="samples before it calibrates (after all bars fill)")
    ap.add_argument("--rational", action="store_true", help="8-coefficient distortion, for a wide lens")
    ap.add_argument("--fisheye", action="store_true", help="equidistant (fisheye) model")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = CameraCalib(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.publish_list(with_entry=False)  # the app's list back to camera_bridge's own
            time.sleep(0.3)
        except Exception:  # Ctrl-C may already have shut the context down
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
