#pragma once

#include <optional>
#include <string>
#include <vector>

#include "quadruped_control/estimator.hpp"

namespace quadruped_control {

// What a driver hands the pipeline each step (TelemetryManager.process_state's arguments). Joints in Isaac order.
struct RawState {
  Vec12 q = Vec12::Zero();
  Vec12 dq = Vec12::Zero();
  Vec4 quat_wxyz = Vec4(1.0, 0.0, 0.0, 0.0);
  Vec3 gyro = Vec3::Zero();
  std::optional<Vec3> accel;    // body-frame specific force; [0, 0, 9.81] when absent
  std::optional<Vec3> pos;      // world position, simulators only
  std::optional<Vec3> vel;      // body-frame ground-truth velocity, simulators only
  std::optional<Vec4> contact;  // FL, FR, RL, RR
};

// StandardState.
struct State {
  Vec4 quat_wxyz = Vec4(1.0, 0.0, 0.0, 0.0);
  Vec3 gyro = Vec3::Zero();
  Vec3 accel = Vec3(0.0, 0.0, 9.81);
  Vec12 q = Vec12::Zero();
  Vec12 dq = Vec12::Zero();
  Vec3 base_pos = Vec3(0.0, 0.0, 0.5);
  bool has_pos = false;
  Vec4 feet_contact = Vec4::Zero();
  Vec3 base_lin_vel = Vec3::Zero();
  Vec3 base_lin_vel_sim = Vec3::Zero();
  Vec3 base_lin_vel_est = Vec3::Zero();
};

enum class LogLevel { Info, Warn, Error };
enum class EventKind { Log, SafetyTrip };

// Side effects for whoever runs the controller: log lines, and a safety trip (banner, ENTER-to-release).
struct Event {
  EventKind kind;
  LogLevel level;
  std::string text;
};

class EventQueue {
 public:
  void log(LogLevel level, std::string text) { events_.push_back({EventKind::Log, level, std::move(text)}); }
  void push(Event e) { events_.push_back(std::move(e)); }
  std::vector<Event> drain() {
    std::vector<Event> out;
    out.swap(events_);
    return out;
  }

 private:
  std::vector<Event> events_;
};

}  // namespace quadruped_control
