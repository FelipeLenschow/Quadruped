#pragma once

#include <atomic>
#include <future>
#include <thread>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "messages.hpp"
#include "realtime.hpp"
#include "robot_io.hpp"

namespace quadruped_real {

// Every policy is trained at 20 ms; the motors want a fresh command every 1-10 ms.
constexpr double kPolicyDt = 0.02;
constexpr double kLowCmdDt = 0.005;
constexpr int kTicksPerPolicy = 4;
// No LowState for this long means the stream has stalled (it normally arrives every 2 ms).
constexpr double kLowStateStale = 0.1;

struct LoopConfig {
  std::string interface;
  std::optional<int> core;
  int priority = 80;
  double kp = 0.0, kd = 0.0;
  double emergency_kd = 5.0, emergency_kd_rear = 5.0;
  double contact_threshold = 10.0;
  double fsr_bias = 16.0;
  double calib_time = 1.0, calib_offset_max = 25.0, calib_spread_max = 3.0;
};

// The thread that owns the robot: LowState in, controller step, LowCmd out. With a reserved core it runs
// there alone at SCHED_FIFO, and so do the SDK's DDS threads; everything else reaches it through mailboxes.
class ControlLoop {
 public:
  ControlLoop(LoopConfig cfg, CommandBox& commands, SnapshotBox& snapshots);
  ~ControlLoop();

  // Starts the thread: core, SDK, motion mode release. Resolves once the SDK is up (or failed).
  std::future<std::string> start();
  // Hands over the controller; the loop runs from here on.
  void run(std::unique_ptr<qc::Controller> controller);
  void stop();
  // After stop() and once nothing else uses the robot: releases the SDK endpoints.
  void close();

  RobotIO& robot() { return robot_; }

 private:
  void thread_main(std::promise<std::string> ready);
  void loop();
  void policy_tick();
  void apply(Command& c, Snapshot& snap);
  bool read_low_state(double now, Snapshot& snap);
  qc::RawState raw_state(Snapshot& snap);
  void collect_fsr_sample(const Eigen::Vector4d& fsr, Snapshot& snap);
  void send(const Eigen::Matrix<float, 12, 1>& targets);
  void fault(const std::string& what);
  void log(qc::LogLevel level, std::string text, Snapshot& snap);

  LoopConfig cfg_;
  CommandBox& commands_;
  SnapshotBox& snapshots_;
  RobotIO robot_;
  std::thread thread_;
  std::atomic<bool> running_{true};
  std::promise<std::unique_ptr<qc::Controller>> controller_promise_;
  std::unique_ptr<qc::Controller> controller_;

  LowState low_state_{};
  uint64_t low_state_seq_ = 0;
  bool have_low_state_ = false;
  double low_state_time_ = 0.0;
  bool low_state_stale_ = false;
  LowCmd low_cmd_;
  std::optional<Eigen::Matrix<float, 12, 1>> last_targets_;

  // Timing over the last stats period, logged with the next snapshot.
  double stats_start_ = 0.0, step_sum_ = 0.0, step_max_ = 0.0, late_max_ = 0.0;
  long step_count_ = 0, skipped_ = 0;

  Eigen::Vector4d fsr_offset_;
  std::vector<Eigen::Vector4d> calib_samples_;
  int calib_n_ = 0;
  bool calib_done_ = true;
};

double steady_now();
double wall_now();

}  // namespace quadruped_real
