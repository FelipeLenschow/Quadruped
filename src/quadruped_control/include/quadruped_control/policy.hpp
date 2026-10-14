#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "quadruped_control/estimator.hpp"
#include "quadruped_control/mlp.hpp"

namespace quadruped_control {

// Observation layouts the policies were trained on (export.py reads which one a checkpoint uses).
enum class ObsLayout { Isaac, Unitree, UnitreeVel, Wtw, UnitreeClock };

ObsLayout layout_from_string(const std::string& name);
std::string to_string(ObsLayout layout);

using Vec4d = Eigen::Vector4d;
using Vec3f = Eigen::Vector3f;

struct WtwConfig {
  std::vector<int> blocks;                        // per-step block widths, gait command and clock last
  VecXf gait_command;                             // 8 or 9 columns
  std::map<std::string, Vec3f> gaits;             // every named gait: phase, offset, bound
  std::vector<std::string> trained;               // the ones this run trained on
  std::map<int, std::pair<float, float>> limits;  // column -> trained range
  double stance = 0.5;                            // duty when the command has no duty column
  std::optional<double> max_stride;
  double yaw_radius = 0.3;
};

struct SpeedClockConfig {
  Vec4d foot_offsets{0.5, 0.0, 0.0, 0.5};
  std::optional<Vec4d> fast_offsets;
  std::optional<Vec4d> fast_swing_times;
  double blend_lo = 0.0, blend_hi = 0.0;
  std::optional<Vec4d> fixed_duty;
  bool reset_to_command = false;
  std::optional<double> reset_speed;
  bool speed_from_measured = false;
  double stride_min = 0.0, stride_gain = 0.0, swing_time = 0.0;
  double min_frequency = 0.0, max_frequency = 0.0, min_duty = 0.0, max_duty = 1.0;
  double filter_time = 1.0;
  std::optional<double> foot_x, foot_y;
  double yaw_radius = 0.0;
  double stand_threshold = 0.0;
  bool integrate_phase = false;
  double sync_gain = 4.0;
  double max_foot_speed = 0.0;
};

struct PolicySpec {
  std::string name;
  std::string source;
  ObsLayout layout = ObsLayout::Isaac;
  int obs_dim = 0;
  int obs_dim_single = 0;
  int history_len = 1;
  float action_scale = 0.25f;
  Eigen::Matrix<float, 12, 1> desired_qpos;
  Mlp mlp;
  std::optional<WtwConfig> wtw;
  std::optional<SpeedClockConfig> clock;
  double trick_clock_hz = 0.0;  // from the run's commands.trick, for policies.json "trick" entries
  int trick_switches = 1;

  // A .qpol file written by quadruped_control.export.
  static PolicySpec from_file(const std::string& path);
};

// What build_obs reads from StandardState. Joints in Isaac order.
struct ObsInput {
  Vec4d quat_wxyz;
  Vec3 lin_vel;
  Vec3 gyro;
  Vec12 q;
  Vec12 dq;
};

// Runtime half of policy_runner.py: observation, history, gait clocks and the network.
class PolicyRunner {
 public:
  explicit PolicyRunner(PolicySpec spec);

  // One policy step; returns the raw actions (targets are actions * action_scale + desired_qpos).
  VecXf infer(const ObsInput& in, const Eigen::VectorXd& commands, double dt = 0.02);
  void reset_history();

  void set_gait_vector(const VecXf& values);
  void set_gait(const std::optional<std::string>& gait, const std::map<std::string, float>& fields = {});

  const PolicySpec& spec() const { return spec_; }
  const VecXf& last_actions() const { return last_actions_; }
  void set_last_actions(const VecXf& actions) { last_actions_ = actions; }
  const VecXf& last_obs() const { return last_obs_; }
  const VecXf& gait_command() const { return gait_command_; }

 private:
  VecXf build_obs(const ObsInput& in, const Eigen::VectorXd& commands);
  VecXf wtw_clock();
  Vec3f limit_foot_speed(const Vec3f& velocity) const;
  double foot_sweep(const Vec3f& velocity) const;
  void reset_speed_clock();
  void clock_timing(Vec4d& offsets, Vec4d& duty, double& f) const;
  Vec4d clock_integrate(const Vec4d& offsets, const Vec4d& d, double f, bool standing, double dt);
  VecXf speed_clock(const Vec3f& velocity, const Vec3f& measured);

  PolicySpec spec_;
  VecXf last_actions_;
  VecXf last_obs_;
  MatXf history_;  // row 0 newest
  bool episode_start_ = true;
  double policy_dt_ = 0.02;

  VecXf gait_command_;
  double gait_index_ = 0.0;

  Vec4d clock_offset_delta_ = Vec4d::Zero();
  Vec4d clock_base_swing_ = Vec4d::Zero();
  Vec4d clock_swing_delta_ = Vec4d::Zero();
  double clock_speed_ = 0.0;
  double clock_index_ = 0.0;
  Vec4d clock_phase_ = Vec4d::Zero();
  bool clock_fresh_ = true;
};

}  // namespace quadruped_control
