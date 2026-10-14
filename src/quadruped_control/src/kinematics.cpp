#include "quadruped_control/kinematics.hpp"

#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace quadruped_control {

namespace {

Mat3 rot_x(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Mat3 r;
  r << 1, 0, 0, 0, c, -s, 0, s, c;
  return r;
}

Mat3 rot_y(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Mat3 r;
  r << c, 0, s, 0, 1, 0, -s, 0, c;
  return r;
}

const char* kLegNames[4] = {"FL", "FR", "RL", "RR"};

}  // namespace

LegGeometry LegGeometry::from_yaml(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("[Kinematics] Cannot read " + path + ": " + e.what());
  }
  const YAML::Node cfg = root["go2_kinematics"];
  LegGeometry g;
  for (int i = 0; i < 4; ++i) {
    const YAML::Node hip = cfg["hip_origins"][kLegNames[i]];
    g.hip_origins[i] = Vec3(hip[0].as<double>(), hip[1].as<double>(), hip[2].as<double>());
    g.thigh_offset_y[i] = cfg["thigh_offset_y"][kLegNames[i]].as<double>();
  }
  g.thigh_length = cfg["thigh_length"].as<double>();
  g.calf_length = cfg["calf_length"].as<double>();
  g.foot_x_offset = cfg["foot_x_offset"].as<double>();
  return g;
}

Vec3 Kinematics::foot_position_body(int leg, const Vec3& q) const {
  const Mat3 r_ab = rot_x(q[0]);
  const Mat3 r_th = r_ab * rot_y(q[1]);
  const Mat3 r_ca = r_th * rot_y(q[2]);
  return g_.hip_origins[leg] + r_ab * Vec3(0.0, g_.thigh_offset_y[leg], 0.0) +
         r_th * Vec3(0.0, 0.0, -g_.thigh_length) +
         r_ca * Vec3(g_.foot_x_offset, 0.0, -g_.calf_length);
}

Mat3 Kinematics::foot_jacobian_body(int leg, const Vec3& q) const {
  const Vec3 p0 = g_.hip_origins[leg];
  const Vec3 z0 = Vec3::UnitX();

  const Mat3 r_ab = rot_x(q[0]);
  const Vec3 p1 = p0 + r_ab * Vec3(0.0, g_.thigh_offset_y[leg], 0.0);
  const Vec3 z1 = r_ab * Vec3::UnitY();

  const Mat3 r_th = r_ab * rot_y(q[1]);
  const Vec3 p2 = p1 + r_th * Vec3(0.0, 0.0, -g_.thigh_length);
  const Vec3 z2 = r_th * Vec3::UnitY();

  const Mat3 r_ca = r_th * rot_y(q[2]);
  const Vec3 foot = p2 + r_ca * Vec3(g_.foot_x_offset, 0.0, -g_.calf_length);

  Mat3 j;
  j.col(0) = z0.cross(foot - p0);
  j.col(1) = z1.cross(foot - p1);
  j.col(2) = z2.cross(foot - p2);
  return j;
}

Mat3 Kinematics::leg_chain_body(int leg, const Vec3& q) const {
  const Mat3 r_ab = rot_x(q[0]);
  const Vec3 thigh = g_.hip_origins[leg] + r_ab * Vec3(0.0, g_.thigh_offset_y[leg], 0.0);
  const Mat3 r_th = r_ab * rot_y(q[1]);
  const Vec3 knee = thigh + r_th * Vec3(0.0, 0.0, -g_.thigh_length);
  const Vec3 foot = knee + r_th * rot_y(q[2]) * Vec3(g_.foot_x_offset, 0.0, -g_.calf_length);
  Mat3 out;
  out.row(0) = thigh.transpose();
  out.row(1) = knee.transpose();
  out.row(2) = foot.transpose();
  return out;
}

}  // namespace quadruped_control
