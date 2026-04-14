import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Quaternion, Vector3
from nav_msgs.msg import Odometry
import numpy as np
import time
import sys
import os
import argparse
import threading

# Add parent directory to sys.path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class Ros2RealBridge(Node):
    """
    Hardware Bridge for Unitree Go2.
    Translates ROS 2 topics to Unitree SDK (UDP) calls.
    """
    def __init__(self, robot_type="go2"):
        super().__init__("ros2_real_bridge")
        self.robot_type = robot_type

        # 1. Initialize Unitree SDK
        try:
            import unitree_legged_sdk as sdk
            self.sdk = sdk
            self.udp = sdk.UDP(sdk.LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.low_state = sdk.LowState()
            self.low_cmd = sdk.LowCmd()
            self.udp.InitCmdData(self.low_cmd)
            print(f"[RealBridge] Unitree SDK Initialized for {robot_type}")
        except ImportError:
            print("[Error] Unitree SDK not found! Real mode will fail.")
            sys.exit(1)

        # 2. ROS Publishers (Hardware -> ROS)
        self.joint_pub = self.create_publisher(JointState, "/sensors/joint_states", 10)
        self.imu_pub   = self.create_publisher(Imu,        "/sensors/imu", 10)
        self.odom_pub  = self.create_publisher(Odometry,   "/odom", 10)

        # 3. ROS Subscriptions (ROS -> Hardware)
        self.create_subscription(JointState, "/commands/joint_commands", self.command_cb, 10)

        # 4. State & Command Buffers
        self.latest_targets = None

        # 5. Map SDK indices to ROS Type-Grouped order
        # SDK (Leg-Grouped): 0-2 (FR), 3-5 (FL), 6-8 (RR), 9-11 (RL)
        # ROS (Type-Grouped): [FL_hip, FR_hip, RL_hip, RR_hip, FL_th, FR_th, RL_th, RR_th, FL_ca, FR_ca, RL_ca, RR_ca]
        self.sdk_to_ros = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        self.ros_to_sdk = np.argsort(self.sdk_to_ros)
        
        # 6. Start SDK Threads
        self.run_bridge = True
        self.sdk_thread = threading.Thread(target=self._sdk_loop, daemon=True)
        self.sdk_thread.start()

    def command_cb(self, msg):
        """Receive joint targets from the AI Controller."""
        if len(msg.position) == 12:
            # Reorder from ROS Type-Grouped back to SDK Leg-Grouped
            self.latest_targets = np.array(msg.position, dtype=np.float32)[self.ros_to_sdk]

    def _sdk_loop(self):
        """Main SDK communication loop running at 500Hz."""
        print("[RealBridge] SDK loop started.")
        while self.run_bridge and rclpy.ok():
            # 1. Receive State from Robot
            self.udp.Recv()
            self.udp.GetRecv(self.low_state)

            # 2. Publish to ROS
            self._publish_telemetry()

            # 3. Send Commands to Robot
            if self.latest_targets is not None:
                for i in range(12):
                    self.low_cmd.motorCmd[i].q = float(self.latest_targets[i])
                    self.low_cmd.motorCmd[i].dq = 0.0
                    self.low_cmd.motorCmd[i].Kp = 25.0 # Match training gains
                    self.low_cmd.motorCmd[i].Kd = 0.5
                    self.low_cmd.motorCmd[i].tau = 0.0
                
                self.udp.SetSend(self.low_cmd)
                self.udp.Send()

            time.sleep(0.002) # 500Hz loop

    def _publish_telemetry(self):
        now = self.get_clock().now().to_msg()

        # Joint States (Map from SDK Leg-Grouped to ROS Type-Grouped)
        js = JointState()
        js.header.stamp = now
        js.name = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        sdk_q = [float(self.low_state.motorState[i].q) for i in range(12)]
        sdk_dq = [float(self.low_state.motorState[i].dq) for i in range(12)]
        sdk_tau = [float(self.low_state.motorState[i].tauEst) for i in range(12)]
        
        js.position = [sdk_q[i] for i in self.sdk_to_ros]
        js.velocity = [sdk_dq[i] for i in self.sdk_to_ros]
        js.effort   = [sdk_tau[i] for i in self.sdk_to_ros]
        self.joint_pub.publish(js)

        # IMU
        imu = Imu()
        imu.header.stamp = now
        q = self.low_state.imu.quaternion
        imu.orientation.w = float(q[0])
        imu.orientation.x = float(q[1])
        imu.orientation.y = float(q[2])
        imu.orientation.z = float(q[3])
        g = self.low_state.imu.gyroscope
        imu.angular_velocity.x = float(g[0])
        imu.angular_velocity.y = float(g[1])
        imu.angular_velocity.z = float(g[2])
        self.imu_pub.publish(imu)

        # Odom (Body Velocity from SDK)
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
        # Unitree SDK usually provides velocity in body frame
        v = self.low_state.velocity
        odom.twist.twist.linear = Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))
        self.odom_pub.publish(odom)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    args = parser.parse_args()

    rclpy.init()
    node = Ros2RealBridge(args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.run_bridge = False
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
