#pragma once

#include <array>
#include <cmath>
#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace quadruped_real {

// Go2 sensor mounts on the base, as in the Unitree Go2 URDF and model.sdf (quadruped_drivers/sensor_mounts.py).
struct Mount {
  std::array<double, 3> xyz, rpy;
};
inline const Mount kRadarMount{{0.28945, 0.0, -0.046825}, {0.0, 2.8782, 0.0}};

// x, y, z, w for roll, pitch, yaw.
inline std::array<double, 4> quaternion(const std::array<double, 3>& rpy) {
  const double cr = std::cos(rpy[0] / 2), sr = std::sin(rpy[0] / 2);
  const double cp = std::cos(rpy[1] / 2), sp = std::sin(rpy[1] / 2);
  const double cy = std::cos(rpy[2] / 2), sy = std::sin(rpy[2] / 2);
  return {sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
          cr * cp * cy + sr * sp * sy};
}

// Rz(yaw) Ry(pitch) Rx(roll), as the transform below.
inline Eigen::Matrix3d matrix(const std::array<double, 3>& rpy) {
  const double cr = std::cos(rpy[0]), sr = std::sin(rpy[0]);
  const double cp = std::cos(rpy[1]), sp = std::sin(rpy[1]);
  const double cy = std::cos(rpy[2]), sy = std::sin(rpy[2]);
  Eigen::Matrix3d m;
  m << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
       sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
       -sp, cp * sr, cp * cr;
  return m;
}

inline geometry_msgs::msg::TransformStamped transform(const std::string& parent, const std::string& child,
                                                      const Mount& mount) {
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = parent;
  t.child_frame_id = child;
  t.transform.translation.x = mount.xyz[0];
  t.transform.translation.y = mount.xyz[1];
  t.transform.translation.z = mount.xyz[2];
  const auto q = quaternion(mount.rpy);
  t.transform.rotation.x = q[0];
  t.transform.rotation.y = q[1];
  t.transform.rotation.z = q[2];
  t.transform.rotation.w = q[3];
  return t;
}

}  // namespace quadruped_real
