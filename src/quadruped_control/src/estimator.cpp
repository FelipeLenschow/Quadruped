#include "quadruped_control/estimator.hpp"

#include <cmath>

namespace quadruped_control {

namespace {
const Vec3 kGravityWorld(0.0, 0.0, -9.81);
}

StateEstimator::StateEstimator(const Kinematics& kinematics, const EstimatorParams& params)
    : kin_(kinematics), params_(params), relax_factor_(std::exp(-params.dt / 0.5)) {
  q_.setZero();
  q_.diagonal() << params.q_vel, params.q_vel, params.q_vel, params.q_bias, params.q_bias,
      params.q_bias;
  f_.setIdentity();
  f_.topRightCorner<3, 3>() = -params.dt * Mat3::Identity();
  reset();
}

void StateEstimator::reset() {
  x_.setZero();
  p_.setIdentity();
  no_contact_time_ = 0.0;
}

Vec3 StateEstimator::update(const Vec4& quat_wxyz, const Vec3& accel_body,
                            const Vec4& feet_contact, const std::optional<Vec12>& joint_pos,
                            const std::optional<Vec12>& joint_vel,
                            const std::optional<Vec3>& gyro_body) {
  const Mat3 r = rot_from_quat(quat_wxyz);

  const bool in_contact = (feet_contact.array() > 0.5).any();
  if (in_contact) {
    no_contact_time_ = 0.0;
  } else {
    no_contact_time_ += params_.dt;
  }
  const bool relaxing = !in_contact && no_contact_time_ > params_.no_contact_relax_time;

  const Vec3 v = x_.head<3>();
  const Vec3 b = x_.tail<3>();
  const Vec3 omega = gyro_body ? *gyro_body : Vec3::Zero();
  const Vec3 a_linear = (accel_body - b) + r.transpose() * kGravityWorld;

  if (relaxing) {
    x_.head<3>() = v * relax_factor_;
  } else {
    x_.head<3>() = v + (a_linear - omega.cross(v)) * params_.dt;
  }
  p_ = f_ * p_ * f_.transpose() + q_;

  if (joint_pos && joint_vel && in_contact) {
    for (int leg = 0; leg < 4; ++leg) {
      if (feet_contact[leg] <= 0.5) continue;
      const Vec3 q_leg((*joint_pos)[leg], (*joint_pos)[leg + 4], (*joint_pos)[leg + 8]);
      const Vec3 dq_leg((*joint_vel)[leg], (*joint_vel)[leg + 4], (*joint_vel)[leg + 8]);
      const Vec3 r_foot = kin_.foot_position_body(leg, q_leg);
      const Mat3 j = kin_.foot_jacobian_body(leg, q_leg);
      const Vec3 z = -j * dq_leg - omega.cross(r_foot);

      // H = [I3 | 0], so H x, H P H^T and P H^T are blocks of x and P.
      const Vec3 y = z - x_.head<3>();
      const Mat3 s = p_.topLeftCorner<3, 3>() + params_.r_meas * Mat3::Identity();
      const Eigen::Matrix<double, 6, 3> k = p_.leftCols<3>() * s.inverse();
      x_ += k * y;
      p_ = (Mat6::Identity() - k * Eigen::Matrix<double, 3, 6>::Identity()) * p_;
    }
  }

  if (relaxing) {
    p_.topLeftCorner<3, 3>() += q_.topLeftCorner<3, 3>();
    p_.bottomRightCorner<3, 3>() += q_.bottomRightCorner<3, 3>();
  }

  if (x_.head<3>().norm() > params_.max_speed) {
    x_.head<3>().setZero();
    p_.topLeftCorner<3, 3>().setIdentity();
    p_.bottomRightCorner<3, 3>().setIdentity();
  }

  return x_.head<3>();
}

}  // namespace quadruped_control
