#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "quadruped_control/config.hpp"
#include "quadruped_control/estimator.hpp"
#include "quadruped_control/policy.hpp"
#include "quadruped_control/pose_generator.hpp"
#include "quadruped_control/safety.hpp"
#include "quadruped_control/state.hpp"

namespace quadruped_control {

enum class CommandMode { Velocity, Zero, Trick };

// A policy's policies.json entry: its checks, what it reads as a command, and for tricks the clock and switches.
struct PolicyEntry {
  uint8_t protections = kAllProtections;
  CommandMode commands = CommandMode::Velocity;
  double clock_hz = 0.0;
  int switches = 0;
};

// command_slew.py.
class CommandSlewLimiter {
 public:
  CommandSlewLimiter(double lin_accel, double lin_decel, double yaw_accel, double yaw_decel);
  Eigen::VectorXd operator()(const Eigen::VectorXd& cmd, double dt);
  void reset() { current_.setZero(); }
  bool enabled() const { return enabled_; }
  void set_enabled(bool on) { enabled_ = on; }

 private:
  Vec3 accel_, decel_, current_ = Vec3::Zero();
  bool enabled_;
};

struct StepResult {
  Vec12 targets;          // latest targets (recomputed on policy steps, held in between)
  double max_torque;      // torque ceiling the driver applies
  bool policy_step;       // targets were recomputed: send them to the robot and publish
  std::optional<std::string> pose_status;  // the pose generator ran: its /pose/status
};

enum class PolicyRequest { Refused, AlreadyActive, Unknown, Ready, NeedsLoad };

// LocomotionPipeline without ROS: telemetry state and estimator, policy selection and switching, pose
// generator, safety arbitration, mode blends. Two clocks come in from outside and are never read here:
// sim_time (blends, poses) and now (heartbeat watchdog, input timeouts). Not thread-safe: the caller
// serializes inputs and steps.
class Controller {
 public:
  explicit Controller(ControllerConfig cfg);

  // ---- policies ----
  void set_catalog(std::vector<std::pair<std::string, PolicyEntry>> catalog);
  // The catalog name whose checkpoint is "main" (requests for it mean main, and it is not listed twice).
  void set_main_alias(const std::optional<std::string>& name) { main_alias_ = name; }
  void add_policy(const std::string& name, PolicySpec spec, const PolicyEntry& entry);
  // A runner built elsewhere (the loader thread), so a control loop only moves it in.
  void add_policy(const std::string& name, PolicyRunner runner, const PolicyEntry& entry);
  void unload_policy(const std::string& name);
  // Policy load done outside (file reading must not run in the control loop): report a failure here.
  void set_load_error(const std::string& name, const std::string& error);
  void set_safety_policy(const std::string& name) { safety_name_ = name; }
  void set_active_policy(const std::optional<std::string>& name) { active_ = name; }
  PolicyRequest request_policy(const std::string& name);

  // ---- inputs ----
  void set_mode(const std::string& mode);
  void pose_command(const std::string& name) { pose_.command_pose(name); }
  void set_pose_interp_duration(double seconds) { pose_.set_interp_duration(seconds); }
  void cmd_vel(double vx, double vy, double wz, double now);
  void trick(const std::vector<double>& values, double now);
  void gait(const VecXf& values);
  void gait_override(const VecXf& values, double now);
  SafetyProcessor& safety() { return safety_; }
  const SafetyProcessor& safety() const { return safety_; }

  // ---- control ----
  StepResult step(const RawState& raw, const std::optional<Eigen::VectorXd>& cmd_vel, double sim_time, double now);
  void reset();
  std::vector<Event> drain_events() { return events_.drain(); }

  // ---- state for /robot_state, telemetry and the drivers ----
  const State& state() const { return state_; }
  const StateEstimator& estimator() const { return estimator_; }
  const Vec12& latest_targets() const { return latest_targets_; }
  const std::string& mode() const { return mode_; }
  void force_mode(const std::string& mode) { mode_ = mode; }
  bool transition_active() const { return transition_active_; }
  void set_transition_active(bool on) { transition_active_ = on; }
  const std::optional<std::string>& active_policy() const { return active_; }
  const std::optional<std::string>& pending_policy() const { return pending_; }
  const std::optional<std::string>& policy_error() const { return policy_error_; }
  const std::string& safety_policy() const { return safety_name_; }
  bool safety_policy_active() const { return safety_elapsed_.has_value() && !safety_.is_numb(); }
  std::vector<std::string> available() const;
  std::vector<std::string> loaded() const;
  bool has_policy(const std::string& name) const { return find(name) != nullptr; }
  PolicyRunner* runner(const std::string& name);
  PolicyEntry entry(const std::string& name) const;
  const PoseGenerator& pose() const { return pose_; }
  const CommandSlewLimiter& slew() const { return slew_; }
  void set_slew_enabled(bool on) { slew_.set_enabled(on); }
  // The velocity command of the last policy step, after the slew limit.
  const Eigen::VectorXd& last_command() const { return last_command_; }
  Eigen::Vector4d current_cmd_vel(double now) const;
  std::optional<double> cmd_age(double now) const;
  std::array<double, 3> trick_command(double now) const;
  bool gait_override_active() const { return gait_override_time_.has_value(); }
  const VecXf* gait_command() const;
  const ControllerConfig& config() const { return cfg_; }
  int decimation() const { return cfg_.decimation; }
  void set_decimation(int d) { cfg_.decimation = d; }
  double policy_dt() const { return cfg_.policy_dt; }
  void set_policy_dt(double dt) { cfg_.policy_dt = dt; }

 private:
  struct Loaded {
    std::string name;
    PolicyRunner runner;
    PolicyEntry entry;
  };

  Loaded* find(const std::string& name);
  const Loaded* find(const std::string& name) const;
  void process_state(const RawState& raw, bool update_estimator);
  Vec12 step_policy(const std::string& name, const Eigen::VectorXd& commands);
  std::vector<double> trick_input(const PolicyEntry& entry, double now);
  void advance_policy_switch();
  void release_gait_override(double now);
  bool is_lying() const;
  void watch_safety_policy();

  ControllerConfig cfg_;
  EventQueue events_;
  StateEstimator estimator_;
  SafetyProcessor safety_;
  PoseGenerator pose_;
  CommandSlewLimiter slew_;
  Vec12 desired_qpos_;
  double lying_cos_;

  std::vector<Loaded> policies_;
  std::vector<std::pair<std::string, PolicyEntry>> catalog_;
  std::map<std::string, std::string> load_errors_;
  std::optional<std::string> main_alias_;
  std::string safety_name_;
  std::optional<std::string> active_, pending_, policy_error_;

  State state_;
  std::string mode_ = "pose";
  long step_counter_ = 0;
  Vec12 latest_targets_;
  std::optional<Vec12> last_measured_q_;
  bool pose_heartbeat_was_ok_ = false;
  bool was_estopped_ = false;
  bool was_numb_ = false;
  bool transition_active_ = false;
  std::optional<double> transition_start_time_;
  double transition_duration_;
  Vec12 transition_start_targets_;
  std::optional<double> slew_time_;
  std::optional<double> safety_elapsed_;
  double lying_for_ = 0.0;

  Eigen::VectorXd last_command_ = Eigen::VectorXd::Zero(4);
  Eigen::Vector4d cmd_ = Eigen::Vector4d::Zero();
  std::optional<double> cmd_time_;
  std::array<double, 3> trick_ = {0.0, 0.0, 0.0};
  std::optional<double> trick_time_;
  double trick_phase_ = 0.0;
  std::optional<VecXf> gait_base_;
  std::optional<double> gait_override_time_;
};

}  // namespace quadruped_control
