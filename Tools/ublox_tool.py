#!/usr/bin/env python3
"""u-blox ZED-F9P RTK setup and checks, over the receiver's USB port (no ROS, no u-center).

    python3 Tools/ublox_tool.py show   [--port /dev/ttyACM0]   # version, base mode, ports, messages, signals
    python3 Tools/ublox_tool.py base   [--port /dev/ttyACM0]   # writes BASE_CONFIG (RAM, BBR and flash)
    python3 Tools/ublox_tool.py rover  [--port /dev/ttyACM0]   # corrections arriving, their age, the satellites

Our RTK pair: two ZED-F9P (firmware HPG 1.32), the corrections from the base's UART2 to the rover's UART2 over
XBee SX 900 MHz radios (XB9X-DMRS-001), 115200 baud on every serial link. The rover is the robot's receiver
(gps_driver reads its USB); the base stands still in the open.

The base surveys its own position at power-up: it averages its fixes for at least SVIN_MIN_DUR seconds and until
their accuracy is under SVIN_ACC_LIMIT, then sends that position (RTCM 1005) with its observations. Until then
the rover has no RTK. Its error shifts every rover position by the same amount, differently each setup: the
limit sets how repeatable GPS positions are between sessions, not whether RTK fixes.

The XBees (read 2026-10-22, the same on both; the base 0013A200 42232393, the rover 0013A200 422322EE): firmware
900A, BD 7 (115200), BR 1 (110 kbps), ID 1985, HP 0, CM 0x555..., PL 2, TO 0x40 (point to multipoint), DL FFFF
(broadcast: no ACK, each packet sent 1 + MT = 4 times), RO 3, NP 256. Each board's second USB ("POWER+XBEE") is the
radio's serial port: AT commands at 115200 (+++ after 1 s of silence). The base's radio never gets that silence
(its F9P sends every second), so it is read over the air from the rover's radio (API mode, remote AT, 0x17).

`show` and `rover` only read. Nothing else may have the port open (stop gps_driver first).
"""

import argparse
import os
import select
import struct
import sys
import termios
import time

# What `base` writes. 2026-10-22: BeiDou MSM4 (1124) added, the rover could not use BeiDou for RTK without it;
# survey-in tightened from 120 s / 2.5 m. The rest was already so: MSM4 GPS, GLONASS, Galileo, 1005, 1230 at 1 Hz on
# UART2, and only RTCM out of UART2 (the radio).
BASE_CONFIG = {
    "CFG-TMODE-MODE": (0x20030001, 1),                       # survey-in
    "CFG-TMODE-SVIN_MIN_DUR": (0x40030010, 300),             # s
    "CFG-TMODE-SVIN_ACC_LIMIT": (0x40030011, 5000),          # 0.1 mm: 0.5 m
    "CFG-UART2-BAUDRATE": (0x40530001, 115200),              # = the XBee's BD (7)
    "CFG-UART2OUTPROT-UBX": (0x10760001, 0),
    "CFG-UART2OUTPROT-NMEA": (0x10760002, 0),
    "CFG-UART2OUTPROT-RTCM3X": (0x10760004, 1),
    "CFG-MSGOUT-RTCM_3X_TYPE1005_UART2": (0x209102bf, 1),    # base position
    "CFG-MSGOUT-RTCM_3X_TYPE1074_UART2": (0x20910360, 1),    # GPS MSM4
    "CFG-MSGOUT-RTCM_3X_TYPE1084_UART2": (0x20910365, 1),    # GLONASS MSM4
    "CFG-MSGOUT-RTCM_3X_TYPE1094_UART2": (0x2091036a, 1),    # Galileo MSM4
    "CFG-MSGOUT-RTCM_3X_TYPE1124_UART2": (0x2091036f, 1),    # BeiDou MSM4
    "CFG-MSGOUT-RTCM_3X_TYPE1230_UART2": (0x20910305, 1),    # GLONASS code-phase biases
}

PORTS = {0x0000: "I2C", 0x0100: "UART1", 0x0201: "UART2", 0x0300: "USB", 0x0400: "SPI"}
PROTOCOLS = {0: "UBX", 1: "NMEA", 2: "RTCM2", 5: "RTCM3"}
GNSS = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 5: "QZSS", 6: "GLO"}
SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}   # by a key's size bits
# Message output keys: each message's I2C key, then UART1, UART2, USB, SPI at +1 .. +4
MESSAGES = {0x209102bd: "RTCM 1005", 0x2091035e: "RTCM 1074", 0x209102cc: "RTCM 1077", 0x20910363: "RTCM 1084",
            0x209102d1: "RTCM 1087", 0x20910368: "RTCM 1094", 0x20910318: "RTCM 1097", 0x2091036d: "RTCM 1124",
            0x209102d6: "RTCM 1127", 0x20910303: "RTCM 1230", 0x209100ba: "NMEA GGA", 0x209100ab: "NMEA RMC",
            0x209100c4: "NMEA GSV", 0x209100bf: "NMEA GSA", 0x209100b0: "NMEA VTG", 0x209100c9: "NMEA GLL",
            0x20910006: "UBX NAV-PVT", 0x20910088: "UBX NAV-SVIN"}
CORRECTION_AGE = ["n/a", "< 1", "1-2", "2-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45-60", "60-90", "90-120",
                  ">= 120"]


class Receiver:
    def __init__(self, port):
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = attrs[1] = attrs[3] = 0   # raw
        attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        self.buf = b""

    @staticmethod
    def frame(cls, mid, payload=b""):
        body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
        a = b = 0
        for x in body:
            a = (a + x) & 0xFF
            b = (b + a) & 0xFF
        return b"\xb5\x62" + body + bytes([a, b])

    def read(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            if select.select([self.fd], [], [], 0.05)[0]:
                try:
                    self.buf += os.read(self.fd, 65536)
                except (BlockingIOError, OSError):
                    pass

    def messages(self):
        out = []
        while True:
            i = self.buf.find(b"\xb5\x62")
            if i < 0:
                self.buf = self.buf[-1:]
                return out
            if len(self.buf) < i + 8:
                self.buf = self.buf[i:]
                return out
            n = struct.unpack_from("<H", self.buf, i + 4)[0]
            if n > 8192:   # a false sync in other data
                self.buf = self.buf[i + 2:]
                continue
            if len(self.buf) < i + 8 + n:
                self.buf = self.buf[i:]
                return out
            out.append((self.buf[i + 2], self.buf[i + 3], self.buf[i + 6:i + 6 + n]))
            self.buf = self.buf[i + 8 + n:]

    def request(self, cls, mid, payload=b"", wait=1.0):
        """Sends a message; returns the payloads of the same class and id that came back, and the ACK (True),
        NAK (False) or None."""
        self.messages()
        os.write(self.fd, self.frame(cls, mid, payload))
        self.read(wait)
        got, ack = [], None
        for c, m, p in self.messages():
            if (c, m) == (cls, mid):
                got.append(p)
            elif c == 0x05 and len(p) >= 2 and (p[0], p[1]) == (cls, mid):
                ack = m == 0x01
        return got, ack

    def get(self, key, layer=0):
        """One configuration value (layer 0 RAM, 1 BBR, 2 flash, 7 default), or None."""
        got, _ = self.request(0x06, 0x8B, struct.pack("<BBHI", 0, layer, 0, key), 0.6)
        if not got:
            return None
        return int.from_bytes(got[0][8:8 + SIZE[(key >> 28) & 7]], "little")

    def group(self, group, layer=0):
        """Every item of a configuration group, {key: value}."""
        out, position = {}, 0
        while True:
            got, _ = self.request(0x06, 0x8B, struct.pack("<BBHI", 0, layer, position, (group << 16) | 0xFFFF), 0.8)
            if not got:
                return out
            p, o, n = got[0], 4, 0
            while o + 4 <= len(p):
                key = struct.unpack_from("<I", p, o)[0]
                size = SIZE[(key >> 28) & 7]
                out[key] = int.from_bytes(p[o + 4:o + 4 + size], "little")
                o += 4 + size
                n += 1
            if n < 64:   # a full page is 64 items
                return out
            position += n

    def set(self, values, layers=0x07):
        """{key: value} into RAM, BBR and flash (layers bits 0, 1, 2). True on ACK."""
        payload = struct.pack("<BBH", 0, layers, 0)
        for key, value in values.items():
            payload += struct.pack("<I", key) + value.to_bytes(SIZE[(key >> 28) & 7], "little")
        _, ack = self.request(0x06, 0x8A, payload, 1.5)
        return ack

    def version(self):
        got, _ = self.request(0x0A, 0x04)
        if not got:
            return None
        p = got[0]
        text = lambda b: b.split(b"\0")[0].decode()
        return ", ".join([text(p[0:30])] + [text(p[k:k + 30]) for k in range(40, len(p), 30)])

    def comms(self):
        """{port: (rx bytes, tx bytes, {protocol: messages in})}."""
        got, _ = self.request(0x0A, 0x36)
        if not got:
            return None
        p = got[0]
        protocols = list(p[4:8])
        out = {}
        for j in range(p[1]):
            o = 8 + 40 * j
            port = PORTS.get(struct.unpack_from("<H", p, o)[0], "?")
            tx = struct.unpack_from("<I", p, o + 4)[0]
            rx = struct.unpack_from("<I", p, o + 12)[0]
            counts = struct.unpack_from("<4H", p, o + 20)
            out[port] = (rx, tx, {PROTOCOLS.get(k, k): c for k, c in zip(protocols, counts) if k != 0xFF})
        return out


def show(rx):
    print("Receiver:", rx.version())
    tmode = rx.group(0x03)
    mode = tmode.get(0x20030001)
    print("Mode:", {0: "rover", 1: "base, survey-in", 2: "base, fixed position"}.get(mode, mode))
    if mode == 1:
        print(f"  survey-in at least {tmode.get(0x40030010)} s, until accuracy <= {tmode.get(0x40030011, 0) / 1e4:.2f} m")
    for name, uart, outprot, inprot in (("UART1", 0x52, 0x74, 0x73), ("UART2", 0x53, 0x76, 0x75)):
        u, o, i = rx.group(uart), rx.group(outprot), rx.group(inprot)
        bit = lambda d, g, item: d.get(0x10000000 | g << 16 | item)
        print(f"{name}: enabled {bit(u, uart, 5)}, {u.get(0x40000000 | uart << 16 | 1)} baud, "
              f"out UBX {bit(o, outprot, 1)} NMEA {bit(o, outprot, 2)} RTCM3 {bit(o, outprot, 4)}, "
              f"in UBX {bit(i, inprot, 1)} NMEA {bit(i, inprot, 2)} RTCM3 {bit(i, inprot, 4)}")
    ports = ["I2C", "UART1", "UART2", "USB", "SPI"]
    print("Messages out (per navigation epoch):")
    for key, rate in sorted(rx.group(0x91).items()):
        if rate:
            label = next((f"{n} -> {ports[key - k]}" for k, n in MESSAGES.items() if 0 <= key - k <= 4), hex(key))
            print(f"  {label}: {rate}")
    rate = rx.group(0x21)
    print(f"Measurement every {rate.get(0x30210001)} ms")
    signals = rx.group(0x31)
    names = {0x10310001: "GPS L1C/A", 0x10310003: "GPS L2C", 0x10310007: "GAL E1", 0x1031000a: "GAL E5b",
             0x1031000d: "BDS B1I", 0x1031000e: "BDS B2I", 0x10310018: "GLO L1", 0x1031001a: "GLO L2"}
    print("Signals:", ", ".join(f"{n} {'on' if signals.get(k) else 'off'}" for k, n in names.items()))
    if mode == 1:
        got, _ = rx.request(0x01, 0x3B)
        if got:
            p = got[0]
            print(f"Survey-in: {struct.unpack_from('<I', p, 8)[0]} s, accuracy {struct.unpack_from('<I', p, 28)[0] / 1e4:.2f} m, "
                  f"{'done' if p[36] else 'running' if p[37] else 'not running'}")


def base(rx):
    print("Receiver:", rx.version())
    ack = rx.set({key: value for key, value in BASE_CONFIG.values()})
    if not ack:
        sys.exit("NAK or no answer: nothing written")
    bad = []
    for name, (key, value) in BASE_CONFIG.items():
        ram, flash = rx.get(key, 0), rx.get(key, 2)
        if ram != value or flash != value:
            bad.append(f"{name}: RAM {ram}, flash {flash}, wanted {value}")
    print("\n".join(bad) if bad else f"Written and read back (RAM and flash): {len(BASE_CONFIG)} settings")


def rover(rx, seconds=5.0):
    print("Receiver:", rx.version())
    before = rx.comms()
    time.sleep(seconds)
    after = rx.comms()
    if before and after:
        for port in after:
            got = {p: after[port][2][p] - before[port][2].get(p, 0) for p in after[port][2]}
            rate = (after[port][0] - before[port][0]) / seconds
            if rate or any(got.values()):
                print(f"{port}: in {rate:.0f} B/s, messages in {seconds:.0f} s: {got}")
    got, _ = rx.request(0x01, 0x07)
    if got:
        p = got[0]
        flags, flags3 = p[21], struct.unpack_from("<H", p, 78)[0]
        print(f"Fix: type {p[20]}, {['no RTK', 'RTK float', 'RTK fixed', '?'][(flags >> 6) & 3]}, "
              f"differential {(flags >> 1) & 1}, {p[23]} satellites, accuracy {struct.unpack_from('<I', p, 40)[0] / 1e3:.3f} m, "
              f"pDOP {struct.unpack_from('<H', p, 76)[0] / 100:.2f}, correction age {CORRECTION_AGE[min((flags3 >> 1) & 15, 12)]} s")
    got, _ = rx.request(0x01, 0x35)
    if got:
        p = got[0]
        sats = []
        for j in range(p[5]):
            o = 8 + 12 * j
            flags = struct.unpack_from("<I", p, o + 8)[0]
            sats.append((GNSS.get(p[o], p[o]), p[o + 2], (flags >> 3) & 1, (flags >> 6) & 1))
        used = [s for s in sats if s[2]]
        print(f"Satellites: {len(sats)} tracked, {len(used)} used, {sum(s[3] for s in sats)} with corrections; "
              f"signal of the used, median {sorted(s[1] for s in used)[len(used) // 2] if used else 0} dBHz "
              f"(under ~35: sky or interference)")
        for g in sorted({s[0] for s in sats}):
            mine = [s for s in sats if s[0] == g]
            print(f"  {g}: {len(mine)} tracked, {sum(s[2] for s in mine)} used, {sum(s[3] for s in mine)} corrected")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["show", "base", "rover"])
    ap.add_argument("--port", default="/dev/ttyACM0")
    args = ap.parse_args()
    rx = Receiver(args.port)
    if not rx.version():
        sys.exit(f"No answer from a u-blox receiver on {args.port}")
    {"show": show, "base": base, "rover": rover}[args.command](rx)


if __name__ == "__main__":
    main()
