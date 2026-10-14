#include "quadruped_control/pose_generator.hpp"

#include <algorithm>
#include <cstdio>

namespace quadruped_control {

PoseGenerator::PoseGenerator(const ControllerConfig& cfg, EventQueue& events)
    : events_(events), library_(cfg.poses), interp_duration_(cfg.pose_interp_duration), duration_(cfg.pose_interp_duration) {
  pushup_low_ << 0.1, -0.1, 0.1, -0.1, 1.4, 1.4, 1.6, 1.6, -2.4, -2.4, -2.4, -2.4;
  pushup_low_ = pushup_low_.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
}

const Vec12* PoseGenerator::pose(const std::string& name) const {
  for (const auto& [n, q] : library_) {
    if (n == name) return &q;
  }
  return nullptr;
}

void PoseGenerator::command_pose(const std::string& name) {
  if (name == "pushup") {
    pushup_ = true;
    pushup_low_phase_ = true;
    pose_name_ = "pushup";
    pending_ = pushup_low_;
    events_.log(LogLevel::Info, "[PoseGenerator] Pushup mode: continuous cycle started");
    return;
  }
  pushup_ = false;
  const Vec12* q = pose(name);
  if (!q) {
    std::string names;
    for (const auto& p : library_) names += (names.empty() ? "" : ", ") + p.first;
    events_.log(LogLevel::Warn, "[PoseGenerator] Unknown pose '" + name + "'. Available: [" + names + "]");
    return;
  }
  pose_name_ = name;
  pending_ = *q;
  char buf[160];
  std::snprintf(buf, sizeof(buf), "[PoseGenerator] Interpolating to '%s' over %.1fs", name.c_str(), interp_duration_);
  events_.log(LogLevel::Info, buf);
}

void PoseGenerator::set_interp_duration(double seconds) {
  if (seconds <= 0.0) return;
  interp_duration_ = seconds;
  char buf[96];
  std::snprintf(buf, sizeof(buf), "[PoseGenerator] Interpolation duration set to %.1fs", seconds);
  events_.log(LogLevel::Info, buf);
}

void PoseGenerator::set_status(double progress) {
  const std::string name = pushup_ ? "pushup (" + std::string(pushup_low_phase_ ? "low" : "high") + ")" : pose_name_;
  char buf[160];
  if (progress >= 1.0) {
    std::snprintf(buf, sizeof(buf), "%s|%.3f|complete", name.c_str(), progress);
  } else {
    std::snprintf(buf, sizeof(buf), "%s|%.3f|%.0f%%", name.c_str(), progress, progress * 100.0);
  }
  status_ = buf;
}

Vec12 PoseGenerator::step(const Vec12& measured_q, double current_time) {
  if (!initialized_) {
    current_ = measured_q.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
    initialized_ = true;
  }
  if (pending_) {
    start_ = current_;
    end_ = *pending_;
    duration_ = interp_duration_;
    start_time_ = current_time;
    pending_.reset();
  }
  if (!start_time_) {
    set_status(1.0);
    return current_;
  }

  const double alpha = std::clamp((current_time - *start_time_) / duration_, 0.0, 1.0);
  current_ = start_ + alpha * (end_ - start_);
  set_status(alpha);

  if (alpha >= 1.0) {
    start_time_.reset();
    if (pushup_) {
      pushup_low_phase_ = !pushup_low_phase_;
      if (pushup_low_phase_) {
        pending_ = pushup_low_;
      } else {
        const Vec12* stand = pose("stand");
        pending_ = stand ? *stand : default_stance();
      }
    }
  }
  return current_;
}

void PoseGenerator::sync_to_current(const std::optional<Vec12>& seed_qpos) {
  if (seed_qpos) {
    current_ = seed_qpos->unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
    initialized_ = true;
  } else {
    initialized_ = false;
  }
  start_time_.reset();
  pending_.reset();
  pushup_ = false;
  pose_name_ = "holding";
}

}  // namespace quadruped_control
