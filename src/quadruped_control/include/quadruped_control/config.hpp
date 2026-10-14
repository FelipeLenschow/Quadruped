#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "quadruped_control/estimator.hpp"

namespace quadruped_control {

// Isaac Lab's init_state.joint_pos (robot_defaults.DEFAULT_STANCE_QPOS), Isaac order.
Vec12 default_stance();

// The checks a policy can ask for in policies.json.
enum Protection : uint8_t { kHeartbeat = 1, kTilt = 2, kPitch = 4, kRom = 8, kAllProtections = 15 };
uint8_t protections_from_names(const std::vector<std::string>& names);
std::vector<std::string> protection_names(uint8_t mask);

// What the pipeline reads from config.yaml, with the same defaults as the Python code.
struct ControllerConfig {
  int decimation = 4;
  double sim_dt = 0.001;
  double policy_dt = 0.004;
  double action_scale = 0.25;
  double saturation = 0.9;
  Vec12 hard_min, hard_max;
  std::array<std::string, 12> joint_names;
  double motor_max_torque = 45.0;
  double watchdog_timeout = 1.0;
  double safety_policy_timeout = 10.0;
  double lying_hold = 1.0;
  double lying_tilt_deg = 20.0;
  double lying_pose_err = 0.3;
  double lying_joint_vel = 1.0;
  Vec12 lie_pose;
  std::vector<std::pair<std::string, Vec12>> poses;  // stand, lie_flat, sit
  double pose_interp_duration = 3.0;
  double policy_switch_blend = 0.5;
  double mode_transition_duration = 3.0;
  double lin_accel = 0.0, lin_decel = 0.0, yaw_accel = 0.0, yaw_decel = 0.0;
  double cmd_timeout = 0.5;
  bool use_estimator = false;
  bool safety_switch = true;
  EstimatorParams estimator;
  LegGeometry geometry;

  ControllerConfig();
  // config.yaml plus the leg geometry yaml; sim_dt and use_estimator come from the driver.
  static ControllerConfig from_yaml(const std::string& config_path, const std::string& geometry_path, double sim_dt,
                                    bool use_estimator, bool safety_switch = true);
};

}  // namespace quadruped_control
