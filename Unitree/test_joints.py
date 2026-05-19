import os
import sys
import time
import math
import argparse
import threading

# Add project root and SDK to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "unitree_sdk2_python"))
)

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Joint Limits from Go2 URDF
JOINT_LIMITS = [
    (-1.0472, 1.0472),   # 0: FR_hip
    (-1.5708, 3.4907),   # 1: FR_thigh
    (-2.7227, -0.83776), # 2: FR_calf
    (-1.0472, 1.0472),   # 3: FL_hip
    (-1.5708, 3.4907),   # 4: FL_thigh
    (-2.7227, -0.83776), # 5: FL_calf
    (-1.0472, 1.0472),   # 6: RR_hip
    (-0.5236, 4.5379),   # 7: RR_thigh
    (-2.7227, -0.83776), # 8: RR_calf
    (-1.0472, 1.0472),   # 9: RL_hip
    (-0.5236, 4.5379),   # 10: RL_thigh
    (-2.7227, -0.83776)  # 11: RL_calf
]

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf"
]

class JointTester:
    def __init__(self, interface=None, min_pct=0.3, max_pct=0.7):
        self.min_pct = min_pct
        self.max_pct = max_pct
        
        # Initialization
        os.environ.pop("CYCLONEDDS_URI", None)
        ChannelFactoryInitialize(0, networkInterface=interface)
        
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self._init_low_cmd()
        
        self.current_q = [0.0] * 12
        self.initial_q = None
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)
        
        print("Waiting for LowState to capture initial pose...")
        while self.initial_q is None:
            time.sleep(0.1)
        print("Initial pose captured.")
        
        self.active_joint = -1
        self.start_time = 0.0
        self.state = "HOLD_INITIAL"
        self.interpolation_start_time = 0.0
        self.running = True
        
        self.thread = threading.Thread(target=self.control_loop)
        self.thread.start()

    def low_state_handler(self, msg: LowState_):
        for i in range(12):
            self.current_q[i] = float(msg.motor_state[i].q)
        if self.initial_q is None:
            self.initial_q = list(self.current_q)

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 2.146e9
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 1.6e4
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def get_target(self, joint_idx, t):
        low, high = JOINT_LIMITS[joint_idx]
        amplitude_pct = (self.max_pct - self.min_pct) / 2.0
        center_pct = (self.max_pct + self.min_pct) / 2.0
        
        # 0.25 Hz = 4 seconds per full sine wave
        freq = 0.25 
        current_pct = center_pct + amplitude_pct * math.sin(2 * math.pi * freq * t)
        return low + current_pct * (high - low)

    def get_center(self, joint_idx):
        low, high = JOINT_LIMITS[joint_idx]
        center_pct = (self.max_pct + self.min_pct) / 2.0
        return low + center_pct * (high - low)

    def control_loop(self):
        while self.running:
            for i in range(12):
                if self.state == "HOLD_INITIAL":
                    target_q = self.initial_q[i]
                elif self.state == "INTERPOLATING":
                    t_total = time.time() - self.interpolation_start_time
                    progress = min(t_total / 5.0, 1.0)
                    target_q = self.initial_q[i] + (self.get_center(i) - self.initial_q[i]) * progress
                elif self.state == "HOLD_CENTER":
                    target_q = self.get_center(i)
                elif self.state == "TESTING":
                    if i == self.active_joint:
                        t = time.time() - self.start_time
                        target_q = self.get_target(i, t)
                    else:
                        target_q = self.get_center(i)
                else:
                    target_q = self.initial_q[i]
                
                self.low_cmd.motor_cmd[i].q = float(target_q)
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 45.0  # Standard Go2 position Kp
                self.low_cmd.motor_cmd[i].kd = 1.0   # Standard Go2 damping Kd
                self.low_cmd.motor_cmd[i].tau = 0.0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)
            time.sleep(0.002) # 500Hz
            
    def stop(self):
        self.running = False
        self.thread.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", type=str, default=None)
    parser.add_argument("--min_pct", type=float, default=0.3, help="Minimum percentage of joint range (default: 0.3)")
    parser.add_argument("--max_pct", type=float, default=0.7, help="Maximum percentage of joint range (default: 0.7)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print(" UNITREE JOINT TESTER ")
    print("="*50)
    print("WARNING: Ensure the robot is SUSPENDED safely! All joints will move to 50% of their range when started.")
    print("="*50)
    
    print("Initializing SDK and waiting for LowState to capture initial pose...")
    tester = JointTester(args.interface, args.min_pct, args.max_pct)
    
    input("\n[STEP 1] Initial pose captured! Press Enter to start 5s interpolation to center pose...")
    tester.interpolation_start_time = time.time()
    tester.state = "INTERPOLATING"
    
    print("Interpolating to center pose... (waiting 5 seconds)")
    time.sleep(5.0) 
    tester.state = "HOLD_CENTER"
    
    input("\n[STEP 2] Center pose reached! Press Enter to begin testing individual joints...")
    tester.state = "TESTING"
    
    try:
        for i in range(12):
            print(f"\n[{i+1}/12] Next joint to test: {JOINT_NAMES[i]}")
            input("Press Enter to start oscillating...")
            
            tester.active_joint = i
            tester.start_time = time.time()
            
            input(f"--> Oscillating {JOINT_NAMES[i]}! Press Enter to stop this joint...")
            tester.active_joint = -1
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        pass
        
    finally:
        tester.stop()
        print("\nTest completed.")

if __name__ == '__main__':
    main()
