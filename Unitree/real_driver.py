import os
import sys
import time
import numpy as np
import argparse
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
import yaml

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "unitree_sdk2_python"))
)

from pipeline import LocomotionPipeline

# SDK2 Imports
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.go2.vui.vui_client import VuiClient


class RealDriver(Node):
    def __init__(self, robot="go2", internal_policy=None, obs_dim=45, interface=None):
        super().__init__("real_driver")
        self.robot_type = robot

        # 1. SDK2 Initialization
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.low_state = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        self.vui = VuiClient()
        self.vui.SetTimeout(5.0)
        self.vui.Init()
        self._current_brightness = -1

        # Release high-level motion mode so low-level control can take over
        self.get_logger().info("[RealDriver] Checking motion mode...")
        sc = SportClient()
        sc.SetTimeout(5.0)
        sc.Init()
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result['name']:
            self.get_logger().warn(
                f"[RealDriver] Robot is in mode '{result['name']}' — calling StandDown + ReleaseMode."
            )
            sc.StandDown()
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1)
        self.get_logger().info("[RealDriver] Motion mode released. Low-level control is active.")

        # 2. Locomotion Pipeline
        try:
            self.pipeline = LocomotionPipeline(
                node=self,
                robot_type=robot,
                checkpoint=internal_policy,
                obs_dim=obs_dim,
                use_estimator=True,  # Usually True on physical hardware
                sim_dt=0.005
            )
        except ImportError:
            self.get_logger().error("[RealDriver] PyTorch not found. Internal policy disabled. Running in TELEMETRY ONLY mode.")
            self.pipeline = LocomotionPipeline(
                node=self,
                robot_type=robot,
                checkpoint=None,
                obs_dim=obs_dim,
                use_estimator=True,
                sim_dt=0.005
            )
        self.pipeline.decimation = 4  # 200 Hz loop / 4 = 50 Hz policy
        self.pipeline.policy_dt = self.pipeline.decimation * self.pipeline.sim_dt

        # 4. Teleop Subscription
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)
        self.cmds_vel = np.zeros(4)  # [vx, vy, wz, height_cmd(unused)]

        # Dynamic gains subscription & initialization
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Configs", "config.yaml"))
        self.kp = 0.0
        self.kd = 0.0
        try:
            with open(config_path, 'r') as f:
                cfg_data = yaml.safe_load(f)
                ctrl_cfg = cfg_data.get("control", {})
                self.kp = float(ctrl_cfg.get("kp", 0.0))
                self.kd = float(ctrl_cfg.get("kd", 0.0))
        except Exception as e:
            self.kp = 0.0
            self.kd = 0.0
            self.get_logger().warn(f"[RealDriver] Could not load gains from config.yaml, defaulting to 0.0: {e}")

        # Damping used when the safety gate cuts the policy. kp=0 with kd=0 leaves
        # the joints completely free and the robot collapses; a damping kd makes
        # it sink under control. The motor applies this itself, so it is bounded
        # by the motor, not by the (zero) safety torque budget.
        self.emergency_kd = 5.0
        # Counts ABOVE each foot's own no-load offset, not an absolute reading.
        # The four FSRs sit at visibly different zeros, so one absolute cut is a
        # different gate on every foot: the foot with the highest offset trips
        # early and the lowest may never trip at all. If the gate is too high the
        # feet never register contact, the estimator never gets its leg-odometry
        # correction, and the velocity estimate diverges without bound - which
        # feeds garbage straight into the policy.
        self.contact_threshold = 10.0
        _fsr_bias = 16.0
        self._calib_time = 1.0
        self._calib_offset_max = 25.0
        self._calib_spread_max = 3.0
        try:
            with open(config_path, 'r') as f:
                _cfg = yaml.safe_load(f) or {}
            self.emergency_kd = float(_cfg.get("safety", {}).get("emergency_kd", 5.0))
            _est = _cfg.get("state_estimator", {}) or {}
            self.contact_threshold = float(_est.get("contact_threshold", 10.0))
            _fsr_bias = float(_est.get("fsr_bias", 16.0))
            self._calib_time = float(_est.get("fsr_calibration_time", 1.0))
            self._calib_offset_max = float(_est.get("fsr_offset_max", 25.0))
            self._calib_spread_max = float(_est.get("fsr_spread_max", 3.0))
        except Exception:
            pass

        # Per-foot no-load offset [FL, FR, RL, RR], measured at startup by
        # _collect_fsr_sample. Until that finishes, the configured bias stands in
        # for all four so contact still works from the first tick.
        self.fsr_offset = np.full(4, _fsr_bias, dtype=np.float64)
        self._calib_samples = []
        self._calib_n = max(0, int(round(self._calib_time / 0.005)))  # 200 Hz loop
        self._calib_done = self._calib_n == 0
        self.get_logger().info(
            f"[RealDriver] Contact threshold: {self.contact_threshold:.0f} counts "
            f"above each foot's offset (start {_fsr_bias:.0f}, "
            + (f"calibrating over {self._calib_time:.1f}s)"
               if not self._calib_done else "calibration disabled)"))
        if not self._calib_done:
            self.get_logger().warn(
                "[RealDriver] Keep the feet UNLOADED for the next "
                f"{self._calib_time:.1f}s - the FSR zero is being measured now.")

        # Raw foot force, so the threshold can be chosen by measuring instead of
        # guessing: watch /sensors/foot_force while loading each foot.
        self.foot_force_pub = self.create_publisher(
            Float32MultiArray, "/sensors/foot_force", 10)
        self._ff_tick = 0

        # The gate itself - [threshold, offset x4] - broadcast at 1 Hz. The twin
        # viewer runs on the operator's laptop, off a DIFFERENT Configs/config.yaml
        # than this one: changing the threshold here and reading it off the screen
        # there showed the laptop's stale value while the robot gated on the new
        # one. The offsets are measured here and cannot be guessed off-robot at
        # all, so nothing off-robot infers either - both are published.
        self.foot_cal_pub = self.create_publisher(
            Float32MultiArray, "/sensors/foot_force_calibration", 10)
        self.create_timer(1.0, self._publish_contact_calibration)

        self.create_subscription(Float32, "/control/kp", self.kp_cb, 10)
        self.create_subscription(Float32, "/control/kd", self.kd_cb, 10)
        self._startup_console_check = True

        # 5. Initialization logic for SDK
        self._init_low_cmd()

        # 6. Control Loop (200Hz to match simulation)
        self.create_timer(0.005, self.control_loop)
        self.get_logger().info(
            f"[RealDriver] Initialized for {robot}. Ready for deployment."
        )

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 2.146e9  # PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 1.6e4  # VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def teleop_cb(self, msg):
        self.cmds_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z, 0.0])

    def kp_cb(self, msg):
        new_kp = float(msg.data)
        if new_kp != self.kp:
            self.kp = new_kp
            self.get_logger().info(f"[RealDriver] Dynamic Kp updated to: {self.kp:.1f}")

    def kd_cb(self, msg):
        new_kd = float(msg.data)
        if new_kd != self.kd:
            self.kd = new_kd
            self.get_logger().info(f"[RealDriver] Dynamic Kd updated to: {self.kd:.2f}")

    def _get_raw_sensor_data(self):
        """Standardizes LowState into raw vectors for the TelemetryManager."""
        raw = self.low_state
        sdk_to_ros = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        q = [float(raw.motor_state[i].q) for i in sdk_to_ros]
        dq = [float(raw.motor_state[i].dq) for i in sdk_to_ros]
        quat = raw.imu_state.quaternion   # [w, x, y, z]
        gyro = raw.imu_state.gyroscope    # body frame
        accel = raw.imu_state.accelerometer # body frame

        # Foot contact from FSR sensors (int16)
        # Unitree LowState foot_force order: [FR, FL, RR, RL] -> reorder to [FL, FR, RL, RR]
        contact = [0.0, 0.0, 0.0, 0.0]
        if hasattr(raw, 'foot_force'):
            ff = raw.foot_force
            fsr = np.array([ff[1], ff[0], ff[3], ff[2]], dtype=np.float64)  # FL,FR,RL,RR

            if not self._calib_done:
                self._collect_fsr_sample(fsr)

            # 200 Hz loop -> publish at 50 Hz. Raw, uncorrected: the offsets go
            # out on /sensors/foot_force_calibration, so a recording keeps what
            # the sensor said and stays comparable with the simulated FSR.
            self._ff_tick += 1
            if self._ff_tick % 4 == 0:
                msg = Float32MultiArray()
                msg.data = [float(v) for v in fsr]
                self.foot_force_pub.publish(msg)

            contact = [float(m) for m in (fsr - self.fsr_offset > self.contact_threshold)]
        # print(q)
        #[0.031, 1.31, -2.85, 0.014, 1.32, -2.83, -0.320, 1.323, -2.83, 0.31, 1.31, -2.80] #Laydown

        return {
            'q': q, 'dq': dq, 'quat': quat, 'gyro': gyro, 'accel': accel, 'contact': contact
        }

    def _publish_contact_calibration(self):
        """Broadcast the FSR gate so viewers show what the robot actually applies."""
        msg = Float32MultiArray()
        msg.data = [float(self.contact_threshold)] + [float(o) for o in self.fsr_offset]
        self.foot_cal_pub.publish(msg)

    def _collect_fsr_sample(self, raw):
        """Average the unloaded FSR readings into a per-foot zero, once, at startup."""
        self._calib_samples.append(np.asarray(raw, dtype=np.float64))
        if len(self._calib_samples) < self._calib_n:
            return

        samples = np.asarray(self._calib_samples)
        mean, spread = samples.mean(axis=0), samples.std(axis=0)
        self._calib_samples = []
        self._calib_done = True

        names = ("FL", "FR", "RL", "RR")
        # A foot taking load during the window reads high and steady; a foot
        # being moved reads noisy. Either way the zero is not a zero, and
        # adopting it would raise that foot's gate by however much it was
        # carrying - so keep the configured bias for that foot and say which.
        bad = (mean > self._calib_offset_max) | (spread > self._calib_spread_max)
        offsets = np.where(bad, self.fsr_offset, mean)
        self.get_logger().info(
            "[RealDriver] FSR zero: "
            + "  ".join(f"{n}={m:.1f}+-{s:.1f}" for n, m, s in zip(names, mean, spread))
            + f"  -> contact above {self.contact_threshold:.0f} counts over each")
        if bad.any():
            rejected = ", ".join(
                f"{n} ({m:.1f}+-{s:.1f})" for n, m, s, b in zip(names, mean, spread, bad) if b
            )
            self.get_logger().warn(
                f"[RealDriver] Rejected FSR zero for {rejected} - loaded or moving "
                f"during calibration (limits: offset<={self._calib_offset_max:.0f}, "
                f"spread<={self._calib_spread_max:.1f}). Kept "
                f"{self.fsr_offset[0]:.0f} from config; restart the driver with the "
                "feet unloaded to measure it.")
        self.fsr_offset = offsets
        self._publish_contact_calibration()

    def control_loop(self):
        """Internal inference logic."""
        if self.low_state is None:
            return

        # Check if console was opened before this pipeline
        if hasattr(self, "_startup_console_check") and self._startup_console_check:
            if not hasattr(self, "_startup_ticks"):
                self._startup_ticks = 0
            self._startup_ticks += 1
            if self._startup_ticks >= 20: # 100ms at 200Hz
                self._startup_console_check = False
                if self.count_publishers("/safety/heartbeat") > 0:
                    self.get_logger().error("[Safety] Console was detected running before driver! Exiting driver for safety.")
                    import sys
                    sys.exit(0)

        raw_data = self._get_raw_sensor_data()

        cmds = self.pipeline.step(
            raw_state_kwargs=raw_data,
            cmd_vel=self.cmds_vel,
            sim_time=time.time()
        )
        
        if "main" in self.pipeline.policy_manager.policies:
            self.send_to_sdk(cmds)

    def send_to_sdk(self, joint_targets):
        """Map ROS Type-Grouped targets to SDK2 motor commands."""
        # Generic map from TelemetryManager: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        # (This matches the order we used in sim_bridges)
        ros_to_sdk = [
            1,
            5,
            9,
            0,
            4,
            8,
            3,
            7,
            11,
            2,
            6,
            10,
        ]

        max_torque = self.pipeline.safety_processor.active_max_torque
        
        for i, ros_idx in enumerate(ros_to_sdk):
            self.low_cmd.motor_cmd[i].q = float(joint_targets[ros_idx])
            self.low_cmd.motor_cmd[i].dq = 0.0
            
            if max_torque <= 0.1:
                # Emergency: damping only, no position hold.
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = self.emergency_kd
                self.low_cmd.motor_cmd[i].tau = 0.0
            else:
                self.low_cmd.motor_cmd[i].kp = self.kp
                self.low_cmd.motor_cmd[i].kd = self.kd
                self.low_cmd.motor_cmd[i].tau = 0.0

        # Set Head Light Brightness based on pipeline mode
        desired_brightness = 0
        if self.pipeline.safety_processor.is_policy_blocked:
            desired_brightness = 10  # Emergency Stop (Max brightness)
        elif self.pipeline.mode == "policy":
            desired_brightness = 0  # Policy Mode (Off)
        elif self.pipeline.mode == "pose":
            desired_brightness = 3  # Pose Generator Mode (Dim)

        if desired_brightness != self._current_brightness:
            self._current_brightness = desired_brightness
            # Run in a background thread to avoid blocking the 200Hz loop
            threading.Thread(target=self.vui.SetBrightness, args=(desired_brightness,), daemon=True).start()

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--interface", type=str, default=None)
    parser.add_argument("--internal_policy", type=str, default=None)
    parser.add_argument("--obs_dim", type=int, default=45)
    args = parser.parse_args()

    # 1. SDK2 Initialization (Must happen BEFORE rclpy.init to claim the DDS domain)
    # Clear ROS 2 config to prevent conflicts with SDK's internal XML
    os.environ.pop("CYCLONEDDS_URI", None)
    try:
        ChannelFactoryInitialize(0, networkInterface=args.interface)
    except Exception as e:
        print(f"[SDK2] Failed to initialize ChannelFactory: {e}")
        sys.exit(1)

    # 2. ROS 2 Initialization (Use a different Domain ID to avoid conflict with SDK)
    with open("Configs/config.yaml", 'r') as f:
        cfg_data = yaml.safe_load(f)
        domain = str(cfg_data.get("network", {}).get("ros_domain_id", "1"))
    os.environ["ROS_DOMAIN_ID"] = domain
    rclpy.init()
    node = RealDriver(args.robot, args.internal_policy, args.obs_dim, interface=args.interface)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
