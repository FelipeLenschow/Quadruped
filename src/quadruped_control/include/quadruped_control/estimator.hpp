#pragma once

#include <optional>

#include <Eigen/Dense>

#include "quadruped_control/kinematics.hpp"

namespace quadruped_control {

using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

// Body -> world rotation from a [w, x, y, z] quaternion.
Mat3 rot_from_quat(const Vec4& quat_wxyz);
// World gravity in the body frame, as Isaac Lab's projected_gravity_b (upright: [0, 0, -9.81]).
Vec3 projected_gravity_b(const Vec4& quat_wxyz);

struct EstimatorParams {
  double dt = 0.02;
  double q_vel = 0.01;
  double q_bias = 1e-3;
  double r_meas = 0.05;
  double max_speed = 5.0;
  double no_contact_relax_time = 0.3;
};

// 6-state LKF, x = [body velocity, accelerometer bias]: IMU prediction, leg-odometry correction.
// Same model as quadruped_core/telemetry/estimator.py, which documents the reasoning.
class StateEstimator {
 public:
  StateEstimator(const Kinematics& kinematics, const EstimatorParams& params = {});

  void reset();

  // Joints in Isaac order (all hips, then thighs, then calves); contact is [FL, FR, RL, RR].
  Vec3 update(const Vec4& quat_wxyz, const Vec3& accel_body, const Vec4& feet_contact,
              const std::optional<Vec12>& joint_pos = std::nullopt,
              const std::optional<Vec12>& joint_vel = std::nullopt,
              const std::optional<Vec3>& gyro_body = std::nullopt);

  Vec3 velocity() const { return x_.head<3>(); }
  Vec3 accel_bias() const { return x_.tail<3>(); }
  const Mat6& covariance() const { return p_; }
  const EstimatorParams& params() const { return params_; }

 private:
  Kinematics kin_;
  EstimatorParams params_;
  double relax_factor_;
  double no_contact_time_ = 0.0;
  Mat6 f_;
  Mat6 q_;
  Vec6 x_;
  Mat6 p_;
};

}  // namespace quadruped_control
