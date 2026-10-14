#pragma once

#include <optional>
#include <string>
#include <utility>

#include "quadruped_control/config.hpp"
#include "quadruped_control/state.hpp"

namespace quadruped_control {

// command_safety_processor.py without ROS: the checks, the trip latch, e-stop and numb.
// The supervisor's parameters and the operator's buttons arrive through the setters.
class SafetyProcessor {
 public:
  SafetyProcessor(const ControllerConfig& cfg, EventQueue& events);

  void heartbeat(double now);
  void set_max_torque_percent(double v);
  void set_base_tilt_limit(double deg);
  void set_forward_tilt_limit(double deg);
  void set_rom_margin(double margin);
  void set_watchdog_timeout(double seconds);  // ignored unless > 0

  // /safety/estop True. Returns true when it newly latched (the caller publishes the state and waits for ENTER).
  bool estop();
  // ENTER at the robot.
  void release_estop();
  void set_numb(const std::string& reason);
  // /safety/numb: true numbs, false releases a numb.
  void numb(bool on);
  // Clears a trip or numb (ENTER after a trip, /safety/reset, a released numb).
  void release();
  // /safety/reset True from the console: releases a trip or numb. Returns whether it did.
  bool safety_reset();
  // Drop the trip latch (mode change to pose; the eval sweeps' stand-in operator keeps the reason).
  void clear_block(bool clear_reason = true);
  // Hold the joint ROM margin at this value whatever the supervisor sends (eval sweeps); nullopt to follow it again.
  void pin_rom_margin(std::optional<double> margin);

  std::pair<bool, std::string> evaluate(const State& state, uint8_t protections, double now) const;
  std::pair<Vec12, double> process(const std::optional<Vec12>& main, const std::optional<Vec12>& safety,
                                   const State& state, uint8_t protections, double now);
  Vec12 clip_soft(const Vec12& targets) const;
  bool heartbeat_ok(double now) const;

  double active_max_torque() const { return (estop_ || numb_) ? 0.0 : active_max_torque_; }
  void set_active_max_torque(double v) { active_max_torque_ = v; }

  bool is_estopped() const { return estop_; }
  bool is_numb() const { return numb_; }
  bool is_policy_blocked() const { return policy_blocked_; }
  bool has_received_heartbeat() const { return has_heartbeat_; }
  double last_heartbeat_time() const { return last_heartbeat_; }
  double watchdog_timeout() const { return watchdog_timeout_; }
  double global_max_torque() const { return global_max_torque_; }
  double max_torque_percent() const { return max_torque_percent_; }
  double rom_margin() const { return rom_margin_; }
  const std::string& trip_reason() const { return trip_reason_; }
  const std::string& numb_reason() const { return numb_reason_; }
  const Vec12& soft_min() const { return soft_min_; }
  const Vec12& soft_max() const { return soft_max_; }
  const Vec12& joint_safe_min() const { return joint_safe_min_; }
  const Vec12& joint_safe_max() const { return joint_safe_max_; }

 private:
  void recompute();

  const ControllerConfig& cfg_;
  EventQueue& events_;
  Vec12 desired_qpos_;
  Vec12 soft_min_, soft_max_;
  Vec12 limits_min_, limits_max_;
  Vec12 joint_safe_min_, joint_safe_max_;

  double max_torque_percent_ = 0.0;
  double base_tilt_limit_ = 30.0;
  double forward_tilt_limit_ = 30.0;
  double rom_margin_ = 0.15;
  std::optional<double> rom_margin_pin_;
  double watchdog_timeout_;
  double tilt_thresh_ = 0.0;
  double global_max_torque_ = 0.0;

  bool shutdown_logged_ = false;
  bool policy_blocked_ = false;
  bool estop_ = false;
  bool numb_ = false;
  std::string numb_reason_;
  std::string trip_reason_;
  double active_max_torque_ = 0.0;
  double last_heartbeat_ = 0.0;
  bool has_heartbeat_ = false;
};

}  // namespace quadruped_control
