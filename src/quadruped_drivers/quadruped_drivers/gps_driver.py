"""u-blox GNSS receiver (the RTK rover) to ROS 2: sensor_msgs/NavSatFix on /gps/fix.

Reads the receiver's serial port (USB: /dev/ttyACM0; config.yaml gps:). The RTK corrections reach the receiver
from the base station over its XBee radio, not through here.

Uses UBX NAV-PVT when the receiver sends it: the receiver's own horizontal and vertical accuracy, and whether the
RTK solution is float or fixed. Without NAV-PVT (for 2 s), NMEA GGA: an accuracy from the fix type and HDOP.

    status: -1 no fix, 0 GNSS fix, 1 differential (SBAS, DGNSS), 2 RTK (float or fixed)
    position_covariance: ENU, diagonal
    altitude: above the WGS84 ellipsoid

The stamp is the time the message was read, system clock: the DLS2 exteroceptive filter compares it with its
state at that time. On DLS2, ros2_translator (gps.sources) passes /gps/fix to sensor_fusion.

/gps/status (std_msgs/String, JSON, 1 Hz; the app's GPS chip): what NavSatFix cannot say. The solution ("RTK
fixed", "RTK float", "DGNSS", "GNSS fix", "no fix", or "no data" when the receiver is silent), accuracies,
satellites, pDOP, the receiver's correction age, and the RTCM corrections it reported over the last second (UBX
RXM-RTCM, which the rover sends on USB): how many arrived, failed their CRC (the radio), were used, their types, and
the seconds since the last good one. A summary goes to the log every SUMMARY_S and on every change of solution.
"""

import argparse
import json
import os
import select
import struct
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import String

from quadruped_common.config_loader import load_config

UBX_SYNC = b"\xb5\x62"
UBX_NAV_PVT = (0x01, 0x07)
UBX_RXM_RTCM = (0x02, 0x32)
MAX_NMEA_LENGTH = 100   # NMEA 0183 allows 82 characters; more without a newline is not NMEA
MAX_UBX_LENGTH = 4096   # a longer payload is a false sync in the binary
PVT_TIMEOUT = 2.0       # [s] without NAV-PVT for this long, GGA is used
SILENT_S = 3.0          # [s] no solution from the receiver for this long: "no data"
SUMMARY_S = 30.0        # [s] between summaries in the log
# NAV-PVT flags3 lastCorrectionAge -> seconds, as text
CORRECTION_AGE = ["n/a", "<1", "1-2", "2-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45-60", "60-90", "90-120",
                  ">=120"]

# GGA fix quality -> (status, label, horizontal std [m] or None: from HDOP, vertical std / horizontal std)
GGA_QUALITY = {
    1: (NavSatStatus.STATUS_FIX, "GNSS fix", None),
    2: (NavSatStatus.STATUS_SBAS_FIX, "DGNSS", None),
    4: (NavSatStatus.STATUS_GBAS_FIX, "RTK fixed", 0.02),
    5: (NavSatStatus.STATUS_GBAS_FIX, "RTK float", 0.3),
}
# HDOP -> horizontal std [m], by quality
GGA_HDOP_STD = {1: 2.5, 2: 1.0}


def ubx_checksum(data):
    a = b = 0
    for byte in data:
        a = (a + byte) & 0xFF
        b = (b + a) & 0xFF
    return bytes((a, b))


def nmea_valid(line):
    """'$...*hh' with the right checksum."""
    star = line.rfind("*")
    if not line.startswith("$") or star < 0 or len(line) < star + 3:
        return False
    checksum = 0
    for c in line[1:star]:
        checksum ^= ord(c)
    try:
        return checksum == int(line[star + 1:star + 3], 16)
    except ValueError:
        return False


class StreamParser:
    """Splits the receiver's byte stream (UBX and NMEA mixed) into checked messages."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data):
        """[("ubx", (cls, id), payload) or ("nmea", line)], the complete ones; the rest waits for more bytes."""
        self.buf += data
        out = []
        while True:
            # the next message start, anything before it is noise
            starts = [i for i in (self.buf.find(b"$"), self.buf.find(UBX_SYNC)) if i >= 0]
            if not starts:
                # keep a last 0xB5: it may be the first half of a sync
                del self.buf[:max(0, len(self.buf) - 1)]
                return out
            del self.buf[:min(starts)]

            if self.buf.startswith(UBX_SYNC):
                if len(self.buf) < 6:
                    return out
                length = struct.unpack_from("<H", self.buf, 4)[0]
                if length > MAX_UBX_LENGTH:
                    del self.buf[:2]
                    continue
                if len(self.buf) < 8 + length:
                    return out
                frame = bytes(self.buf[:8 + length])
                if ubx_checksum(frame[2:6 + length]) == frame[6 + length:]:
                    out.append(("ubx", (frame[2], frame[3]), frame[6:6 + length]))
                    del self.buf[:8 + length]
                else:
                    del self.buf[:2]   # not a frame: look past this sync
                continue

            end = self.buf.find(b"\n", 0, MAX_NMEA_LENGTH + 2)
            # a UBX frame before the newline: this line was cut, the frame is kept
            sync = self.buf.find(UBX_SYNC, 1, end if end >= 0 else MAX_NMEA_LENGTH + 2)
            if sync >= 0:
                del self.buf[:sync]
                continue
            if end < 0:
                if len(self.buf) > MAX_NMEA_LENGTH:
                    del self.buf[:1]
                    continue
                return out
            line = self.buf[:end].decode("ascii", errors="replace").strip()
            del self.buf[:end + 1]
            if nmea_valid(line):
                out.append(("nmea", line))


def parse_nav_pvt(payload):
    """UBX NAV-PVT -> fix dict, or None."""
    if len(payload) < 92:
        return None
    fix_type, flags, _, num_sv = payload[20], payload[21], payload[22], payload[23]
    lon, lat, height, _ = struct.unpack_from("<iiii", payload, 24)
    h_acc, v_acc = struct.unpack_from("<II", payload, 40)
    gnss_fix_ok = bool(flags & 0x01)
    diff_soln = bool(flags & 0x02)
    carr_soln = (flags >> 6) & 0x03   # 0 none, 1 float, 2 fixed

    if not gnss_fix_ok or fix_type not in (2, 3, 4):   # 0 none, 1 dead reckoning, 5 time only
        status, label = NavSatStatus.STATUS_NO_FIX, "no fix"
    elif carr_soln == 2:
        status, label = NavSatStatus.STATUS_GBAS_FIX, "RTK fixed"
    elif carr_soln == 1:
        status, label = NavSatStatus.STATUS_GBAS_FIX, "RTK float"
    elif diff_soln:
        status, label = NavSatStatus.STATUS_SBAS_FIX, "DGNSS"
    else:
        status, label = NavSatStatus.STATUS_FIX, "GNSS fix"
    p_dop = struct.unpack_from("<H", payload, 76)[0]
    flags3 = struct.unpack_from("<H", payload, 78)[0]
    return {
        "status": status, "label": label, "satellites": num_sv,
        "lat": lat * 1e-7, "lon": lon * 1e-7, "alt": height * 1e-3,
        "h_std": h_acc * 1e-3, "v_std": v_acc * 1e-3,
        "covariance_type": NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN,
        "pdop": p_dop * 0.01, "correction_age": CORRECTION_AGE[min((flags3 >> 1) & 0x0F, len(CORRECTION_AGE) - 1)],
    }


def parse_rxm_rtcm(payload):
    """UBX RXM-RTCM (one per RTCM message the receiver got) -> (type, crc failed, used: True/False/None), or None."""
    if len(payload) < 8:
        return None
    flags = payload[1]
    used = {2: True, 1: False}.get((flags >> 1) & 0x03)
    return struct.unpack_from("<H", payload, 6)[0], bool(flags & 0x01), used


def ddm_to_degrees(value, hemisphere):
    """NMEA (d)ddmm.mmmm -> degrees."""
    dot = value.find(".")
    degrees = float(value[:dot - 2]) + float(value[dot - 2:]) / 60.0
    return -degrees if hemisphere in ("S", "W") else degrees


def parse_gga(line):
    """NMEA GGA -> fix dict, or None (not GGA)."""
    fields = line[:line.rfind("*")].split(",")
    if not fields[0].endswith("GGA") or len(fields) < 12:
        return None
    try:
        quality = int(fields[6] or 0)
        satellites = int(fields[7] or 0)
        if quality not in GGA_QUALITY or not fields[2] or not fields[4]:
            return {"status": NavSatStatus.STATUS_NO_FIX, "label": "no fix", "satellites": satellites}
        status, label, h_std = GGA_QUALITY[quality]
        hdop = float(fields[8] or 99.0)
        if h_std is None:
            h_std = hdop * GGA_HDOP_STD[quality]
        return {
            "status": status, "label": label, "satellites": satellites,
            "lat": ddm_to_degrees(fields[2], fields[3]), "lon": ddm_to_degrees(fields[4], fields[5]),
            # GGA's altitude is above the geoid: add the geoid separation for the ellipsoid
            "alt": float(fields[9] or 0.0) + float(fields[11] or 0.0),
            "h_std": h_std, "v_std": 2.0 * h_std,
            "covariance_type": NavSatFix.COVARIANCE_TYPE_APPROXIMATED,
        }
    except (ValueError, IndexError):
        return None


def open_serial(port, baud):
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    tty.setraw(fd)
    attrs = termios.tcgetattr(fd)
    speed = getattr(termios, f"B{baud}")
    attrs[4] = attrs[5] = speed
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd


class GpsDriver(Node):
    def __init__(self, args):
        super().__init__("gps_driver")
        self.args = args
        self.pub = self.create_publisher(NavSatFix, args.topic, 10)
        self.status_pub = self.create_publisher(String, args.status_topic, 10)
        self.parser = StreamParser()
        self.last_pvt = -1e9
        self.label = None
        # Shared with the status timer
        self.lock = threading.Lock()
        self.fix = None                  # the last solution
        self.fix_time = -1e9
        self.rtcm = self.new_window()     # the last second's corrections
        self.summary = self.new_window()  # since the last log summary
        self.last_good_rtcm = None
        self.summary_start = time.monotonic()
        self.create_timer(1.0, self.publish_status)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def read_loop(self):
        fd = None
        warned = False
        while not self.stop.is_set():
            if fd is None:
                try:
                    fd = open_serial(self.args.port, self.args.baud)
                    self.get_logger().info(f"Reading {self.args.port}, publishing {self.args.topic}")
                    warned = False
                except OSError as e:
                    if not warned:
                        self.get_logger().warn(f"Cannot open {self.args.port} ({e.strerror}), retrying")
                        warned = True
                    self.stop.wait(2.0)
                    continue
            try:
                ready, _, _ = select.select([fd], [], [], 0.5)
                data = os.read(fd, 4096) if ready else b""
                if ready and not data:
                    raise OSError("the port closed")
            except OSError as e:
                self.get_logger().warn(f"{self.args.port}: {e}, reopening")
                os.close(fd)
                fd = None
                self.parser = StreamParser()
                continue
            for message in self.parser.feed(data):
                self.on_message(message)
        if fd is not None:
            os.close(fd)

    @staticmethod
    def new_window():
        return {"received": 0, "crc_failed": 0, "used": 0, "types": set()}

    def on_rtcm(self, rtcm, now):
        msg_type, crc_failed, used = rtcm
        with self.lock:
            for window in (self.rtcm, self.summary):
                window["received"] += 1
                window["crc_failed"] += crc_failed
                window["used"] += used is True
                if not crc_failed:
                    window["types"].add(msg_type)
            if not crc_failed:
                self.last_good_rtcm = now

    def on_message(self, message):
        now = time.monotonic()
        if message[0] == "ubx" and message[1] == UBX_RXM_RTCM:
            rtcm = parse_rxm_rtcm(message[2])
            if rtcm is not None:
                self.on_rtcm(rtcm, now)
        elif message[0] == "ubx" and message[1] == UBX_NAV_PVT:
            fix = parse_nav_pvt(message[2])
            if fix is not None:
                self.last_pvt = now
                self.publish(fix)
        elif message[0] == "nmea" and now - self.last_pvt > PVT_TIMEOUT:
            fix = parse_gga(message[1])
            if fix is not None:
                self.publish(fix)

    def publish(self, fix):
        with self.lock:
            self.fix = fix
            self.fix_time = time.monotonic()
        if fix["label"] != self.label:
            self.label = fix["label"]
            self.get_logger().info(self.describe(fix, None, None))

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.status.status = fix["status"]
        msg.status.service = NavSatStatus.SERVICE_GPS
        if fix["status"] == NavSatStatus.STATUS_NO_FIX:
            msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        else:
            msg.latitude, msg.longitude, msg.altitude = fix["lat"], fix["lon"], fix["alt"]
            h, v = fix["h_std"] ** 2, fix["v_std"] ** 2
            msg.position_covariance = [h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, v]
            msg.position_covariance_type = fix["covariance_type"]
        self.pub.publish(msg)

    @staticmethod
    def describe(fix, window, seconds):
        """One log line: the solution, and the corrections over `seconds` if a window is given."""
        text = fix["label"]
        if "h_std" in fix and fix["status"] != NavSatStatus.STATUS_NO_FIX:
            text += f", {fix['satellites']} satellites, {fix['h_std']:.3f} m horizontal"
            if "pdop" in fix:
                text += f", pDOP {fix['pdop']:.1f}, correction age {fix['correction_age']} s"
        if window is not None:
            if window["received"]:
                text += (f"; corrections {window['received'] / seconds:.1f}/s, {window['crc_failed']} CRC failed, "
                         f"{window['used']} used, types {' '.join(map(str, sorted(window['types'])))}")
            else:
                text += "; no corrections"
        return text

    def publish_status(self):
        now = time.monotonic()
        with self.lock:
            fix = self.fix if now - self.fix_time < SILENT_S else None
            window, self.rtcm = self.rtcm, self.new_window()
            last_good = self.last_good_rtcm
            summary = None
            if now - self.summary_start >= SUMMARY_S:
                summary, seconds = self.summary, now - self.summary_start
                self.summary, self.summary_start = self.new_window(), now
        status = {"fix": fix["label"] if fix else "no data", "satellites": fix["satellites"] if fix else 0}
        # Without a fix the receiver's accuracies are meaningless (2^32 mm)
        if fix and "h_std" in fix and fix["status"] != NavSatStatus.STATUS_NO_FIX:
            status.update({"h_acc": round(fix["h_std"], 3), "v_acc": round(fix["v_std"], 3)})
        if fix and "pdop" in fix:
            status.update({"pdop": round(fix["pdop"], 2), "correction_age": fix["correction_age"]})
        status["rtcm"] = {
            "received": window["received"], "crc_failed": window["crc_failed"], "used": window["used"],
            "types": sorted(window["types"]),
            "last_s": round(now - last_good, 1) if last_good is not None else None,
        }
        self.status_pub.publish(String(data=json.dumps(status)))
        if summary is not None:
            self.get_logger().info(self.describe(fix, summary, seconds) if fix else
                                   f"no solution from the receiver for {SILENT_S:.0f} s")

    def destroy_node(self):
        self.stop.set()
        self.thread.join(timeout=2.0)
        super().destroy_node()


def main():
    gps = load_config().get("gps", {})
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=gps.get("port", "/dev/ttyACM0"))
    parser.add_argument("--baud", type=int, default=gps.get("baud", 115200))
    parser.add_argument("--topic", default=gps.get("topic", "/gps/fix"))
    parser.add_argument("--frame_id", default=gps.get("frame_id", "gps"))
    parser.add_argument("--status_topic", default=gps.get("status_topic", "/gps/status"))
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = GpsDriver(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
