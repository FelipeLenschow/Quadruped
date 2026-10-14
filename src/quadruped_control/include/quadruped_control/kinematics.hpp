#pragma once

#include <array>
#include <string>

#include <Eigen/Dense>

namespace quadruped_control {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

// Legs FL=0, FR=1, RL=2, RR=3; joints per leg [hip abduction (X), thigh (Y), calf (Y)].
struct LegGeometry {
  std::array<Vec3, 4> hip_origins;
  std::array<double, 4> thigh_offset_y;
  double thigh_length;
  double calf_length;
  double foot_x_offset;

  // The go2_kinematics block of quadruped_description/config/go2_kinematics.yaml.
  static LegGeometry from_yaml(const std::string& path);
};

class Kinematics {
 public:
  explicit Kinematics(const LegGeometry& geometry) : g_(geometry) {}

  Vec3 foot_position_body(int leg, const Vec3& q_leg) const;
  Mat3 foot_jacobian_body(int leg, const Vec3& q_leg) const;
  // Rows: thigh joint, knee, foot.
  Mat3 leg_chain_body(int leg, const Vec3& q_leg) const;

  const LegGeometry& geometry() const { return g_; }

 private:
  LegGeometry g_;
};

}  // namespace quadruped_control
