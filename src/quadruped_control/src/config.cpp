#include "quadruped_control/config.hpp"

#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace quadruped_control {

namespace {

const std::array<const char*, 4> kProtectionNames = {"heartbeat", "tilt", "pitch", "rom"};

float f32(double v) { return static_cast<float>(v); }

template <typename T>
T get(const YAML::Node& n, const char* key, T fallback) {
  if (!n || !n.IsMap() || !n[key] || n[key].IsNull()) return fallback;
  return n[key].as<T>();
}

Vec12 vec12(const std::vector<double>& v) {
  if (v.size() != 12) throw std::runtime_error("[Config] A pose needs 12 values.");
  Vec12 out;
  for (int i = 0; i < 12; ++i) out[i] = f32(v[i]);
  return out;
}

}  // namespace

Vec12 default_stance() {
  Vec12 q;
  q << 0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5;
  return q.unaryExpr([](double v) { return static_cast<double>(f32(v)); });
}

uint8_t protections_from_names(const std::vector<std::string>& names) {
  uint8_t mask = 0;
  for (const auto& n : names) {
    bool known = false;
    for (size_t i = 0; i < kProtectionNames.size(); ++i) {
      if (n == kProtectionNames[i]) {
        mask |= static_cast<uint8_t>(1u << i);
        known = true;
      }
    }
    if (!known) throw std::invalid_argument("unknown protection '" + n + "'");
  }
  return mask;
}

std::vector<std::string> protection_names(uint8_t mask) {
  std::vector<std::string> out;
  for (size_t i = 0; i < kProtectionNames.size(); ++i) {
    if (mask & (1u << i)) out.emplace_back(kProtectionNames[i]);
  }
  return out;
}

ControllerConfig::ControllerConfig() {
  joint_names = {"FL_hip_joint",   "FR_hip_joint",   "RL_hip_joint",   "RR_hip_joint",
                 "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
                 "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint"};
  for (int i = 0; i < 4; ++i) {
    hard_min[i] = f32(-1.047), hard_max[i] = f32(1.047);
    hard_min[4 + i] = f32(-1.571), hard_max[4 + i] = f32(3.491);
    hard_min[8 + i] = f32(-2.723), hard_max[8 + i] = f32(-0.5);
  }
  lie_pose = default_stance();
  poses = {{"stand", vec12({0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5})},
           {"lie_flat", Vec12::Zero()},
           {"sit", vec12({0.1, -0.1, 0.1, -0.1, 1.2, 1.2, 2.0, 2.0, -2.0, -2.0, -2.5, -2.5})}};
}

ControllerConfig ControllerConfig::from_yaml(const std::string& config_path, const std::string& geometry_path,
                                             double sim_dt, bool use_estimator, bool safety_switch) {
  ControllerConfig c;
  YAML::Node root;
  if (std::ifstream(config_path).good()) root = YAML::LoadFile(config_path);

  const YAML::Node ctrl = root["control"];
  c.decimation = get<int>(ctrl, "decimation", 4);
  c.sim_dt = sim_dt;
  c.policy_dt = c.decimation * sim_dt;
  c.action_scale = get<double>(ctrl, "action_scale", 0.25);
  c.saturation = get<double>(ctrl, "saturation_limit", 0.9);

  const YAML::Node limits = root["joint_limits"];
  auto limit = [&](const char* group, double lo, double hi) -> std::pair<double, double> {
    const YAML::Node g = limits && limits.IsMap() ? limits[group] : YAML::Node();
    return {get<double>(g, "min", lo), get<double>(g, "max", hi)};
  };
  const auto abd = limit("abduction", -1.047, 1.047);
  const auto hip = limit("hip_front", -1.571, 3.491);
  const auto knee = limit("knee", -2.723, -0.5);
  for (int i = 0; i < 4; ++i) {
    c.hard_min[i] = f32(abd.first), c.hard_max[i] = f32(abd.second);
    c.hard_min[4 + i] = f32(hip.first), c.hard_max[4 + i] = f32(hip.second);
    c.hard_min[8 + i] = f32(knee.first), c.hard_max[8 + i] = f32(knee.second);
  }

  c.motor_max_torque = get<double>(root["motor"], "max_torque", 45.0);

  const YAML::Node safety = root["safety"];
  c.watchdog_timeout = get<double>(safety, "watchdog_timeout", 1.0);
  c.safety_policy_timeout = get<double>(safety, "safety_policy_timeout_s", 10.0);
  c.lying_hold = get<double>(safety, "lying_hold_s", 1.0);
  c.lying_tilt_deg = get<double>(safety, "lying_tilt_deg", 20.0);
  c.lying_pose_err = get<double>(safety, "lying_pose_err", 0.3);
  c.lying_joint_vel = get<double>(safety, "lying_joint_vel", 1.0);

  const YAML::Node poses = root["poses"];
  c.pose_interp_duration = get<double>(poses, "default_interp_duration", 3.0);
  for (auto& [name, q] : c.poses) {
    if (poses && poses.IsMap() && poses[name]) q = vec12(poses[name].as<std::vector<double>>());
  }
  if (poses && poses.IsMap() && poses["lie_flat"]) c.lie_pose = vec12(poses["lie_flat"].as<std::vector<double>>());

  c.policy_switch_blend = get<double>(root, "policy_switch_blend_s", 0.5);
  const YAML::Node slew = root["command_slew"];
  c.lin_accel = get<double>(slew, "lin_accel", 0.0);
  c.lin_decel = get<double>(slew, "lin_decel", 0.0);
  c.yaw_accel = get<double>(slew, "yaw_accel", 0.0);
  c.yaw_decel = get<double>(slew, "yaw_decel", 0.0);
  c.cmd_timeout = get<double>(root["cmd_vel"], "timeout", 0.5);

  c.use_estimator = use_estimator;
  c.safety_switch = safety_switch;
  const YAML::Node est = root["state_estimator"];
  const YAML::Node kal = est && est.IsMap() ? est["kalman"] : YAML::Node();
  c.estimator.dt = get<double>(est, "dt", 0.02);
  c.estimator.q_vel = get<double>(kal, "process_noise_vel", 0.01);
  c.estimator.q_bias = get<double>(kal, "process_noise_bias", 1e-3);
  c.estimator.r_meas = get<double>(kal, "measurement_noise", 0.05);
  c.geometry = LegGeometry::from_yaml(geometry_path);
  return c;
}

}  // namespace quadruped_control
