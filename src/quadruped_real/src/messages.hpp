#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <quadruped_control/controller.hpp>

namespace quadruped_real {

namespace qc = quadruped_control;

// Lock-free for its one consumer; producers on the ROS side share a mutex, so the control loop never waits.
template <typename T, size_t N>
class Mailbox {
 public:
  bool push(T value) {
    std::lock_guard<std::mutex> lock(producers_);
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next = (head + 1) % N;
    if (next == tail_.load(std::memory_order_acquire)) return false;
    slots_[head] = std::move(value);
    head_.store(next, std::memory_order_release);
    return true;
  }

  bool pop(T& out) {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) return false;
    out = std::move(slots_[tail]);
    slots_[tail] = T{};
    tail_.store((tail + 1) % N, std::memory_order_release);
    return true;
  }

 private:
  std::array<T, N> slots_{};
  std::atomic<size_t> head_{0}, tail_{0};
  std::mutex producers_;
};

// ROS side -> control loop. `now` is the steady clock when it arrived (input timeouts, the watchdog).
struct Command {
  enum class Kind {
    None, Mode, PoseCommand, PoseDuration, CmdVel, Trick, Gait, GaitOverride, Heartbeat, MaxTorque, TiltLimit,
    ForwardTiltLimit, RomMargin, WatchdogTimeout, SafetyReset, Estop, Numb, ReleaseEstop, ReleaseTrip,
    RequestPolicy, AddPolicy, LoadError, Kp, Kd,
  };
  Kind kind = Kind::None;
  std::string text, detail;
  double value = 0.0;
  bool flag = false;
  std::vector<double> values;
  double now = 0.0;
  std::shared_ptr<qc::PolicyRunner> runner;  // AddPolicy, built on the loader thread
  qc::PolicyEntry entry;
};

// Control loop -> ROS side, once per policy step.
struct Snapshot {
  bool has_state = false;  // false until the first LowState: only events and flags are valid
  double sim_time = 0.0;   // steady clock the controller ran on
  double wall_time = 0.0;  // system clock, for message stamps
  qc::State state;
  qc::Vec12 targets = qc::Vec12::Zero();
  double max_torque = 0.0;
  double global_max_torque = 0.0;
  std::vector<qc::Event> events;
  std::vector<std::string> load_requests;
  std::optional<std::string> pose_status;
  std::string mode, pose_name;
  bool transition = false, pose_done = false;
  bool estop = false, numb = false, blocked = false, safety_policy_active = false;
  std::string trip_reason, numb_reason;
  std::optional<std::string> active, pending, error;
  std::vector<std::string> available, protections;
  int trick_switches = 0;
  std::array<double, 3> trick{};
  std::optional<Eigen::VectorXf> gait;
  bool gait_override = false;
  Eigen::Vector3d command = Eigen::Vector3d::Zero();
  std::optional<Eigen::Vector4d> foot_force;  // raw FSR [FL, FR, RL, RR]
  Eigen::Vector4d fsr_offset = Eigen::Vector4d::Zero();
  double contact_threshold = 0.0;
  bool calibration_changed = false;
};

using CommandBox = Mailbox<Command, 512>;
using SnapshotBox = Mailbox<std::unique_ptr<Snapshot>, 128>;

}  // namespace quadruped_real
