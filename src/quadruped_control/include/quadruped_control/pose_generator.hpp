#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "quadruped_control/config.hpp"
#include "quadruped_control/state.hpp"

namespace quadruped_control {

// pose_generator.py: time-based interpolation between named poses, and the pushup cycle.
class PoseGenerator {
 public:
  PoseGenerator(const ControllerConfig& cfg, EventQueue& events);

  void command_pose(const std::string& name);
  void set_interp_duration(double seconds);  // ignored unless > 0
  Vec12 step(const Vec12& measured_q, double current_time);
  // Hold seed_qpos, or with none re-read the measured joints on the next step (torque off).
  void sync_to_current(const std::optional<Vec12>& seed_qpos = std::nullopt);

  bool is_active() const { return start_time_.has_value(); }
  bool is_complete() const { return initialized_ && !start_time_; }
  const std::string& current_pose_name() const { return pose_name_; }
  // "name|progress|label" as last published on /pose/status.
  const std::string& status() const { return status_; }
  double interp_duration() const { return interp_duration_; }

 private:
  const Vec12* pose(const std::string& name) const;
  void set_status(double progress);

  EventQueue& events_;
  std::vector<std::pair<std::string, Vec12>> library_;
  Vec12 pushup_low_;
  double interp_duration_;
  Vec12 start_ = Vec12::Zero(), end_ = Vec12::Zero(), current_ = Vec12::Zero();
  std::optional<double> start_time_;
  std::optional<Vec12> pending_;
  double duration_;
  std::string pose_name_ = "none";
  bool initialized_ = false;
  bool pushup_ = false;
  bool pushup_low_phase_ = false;
  std::string status_;
};

}  // namespace quadruped_control
