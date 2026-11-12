// A MuJoCo Go2 on the robot's own low-level DDS topics, to run the real driver against on a PC.
//
// rt/lowstate goes out at the physics rate (joints, IMU, foot force in raw FSR counts); rt/lowcmd comes in and
// is applied as the motors do it, tau = kp (q* - q) + kd (dq* - dq) + tau_ff, after the same CRC check the
// robot makes (a command with a bad CRC is ignored). Joints are in the SDK's order: FR, FL, RR, RL,
// hip/thigh/calf each. Use the loopback interface on both sides (--interface lo here and on the driver) so
// nothing reaches a real robot.
//
// The L1 lidar goes out as the robot sends it (--no_lidar: not): rt/utlidar/cloud, about 4200 points at 15.4 Hz
// in the lidar's frame, each at its own time, with ranges real_lidar.range_offset too long, hitting the robot's own
// body; and rt/utlidar/imu, its gyro in the lidar's axes. real_sensors reads them as on the robot.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>
#include <rclcpp/utilities.hpp>
#include <unitree/idl/ros2/Imu_.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#include "config_loader.hpp"
#include "robot_io.hpp"

using namespace quadruped_real;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using Clock = std::chrono::steady_clock;

namespace {

const char* kLegs[4] = {"FR", "FL", "RR", "RL"};
const char* kParts[3] = {"hip", "thigh", "calf"};
// poses.lie_flat in config.yaml, per leg in kLegs order: hip, thigh, calf.
const double kLie[4][3] = {{-0.45, 1.25, -2.6}, {0.45, 1.25, -2.6}, {-0.65, 1.25, -2.6}, {0.65, 1.25, -2.6}};
constexpr double kFsrBias = 16.0, kFsrScale = 2.66;  // raw counts with the foot unloaded; newtons per count

int id(const mjModel* m, mjtObj type, const std::string& name) {
  const int i = mj_name2id(m, type, name.c_str());
  if (i < 0) throw std::runtime_error("no '" + name + "' in the model");
  return i;
}

builtin_interfaces::msg::dds_::Time_ wall_stamp(double ago) {
  const double t = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() - ago;
  return builtin_interfaces::msg::dds_::Time_(static_cast<int32_t>(t), static_cast<uint32_t>((t - std::floor(t)) * 1e9));
}

// The pattern of quadruped_drivers/l1_lidar.py (fitted to a real recording): a beam on a mirror turning once per
// kPeriod along a great circle kTilt off the lidar's z axis, its plane turning about z at kAzRate, a return every
// kPeriod / kSlots on the half above the lidar's xy plane; kCloudSlots slots per cloud. Rays start kMinRange out,
// and hits on the base nearer than kHousing are dropped (the lidar sits inside the head's mesh).
class L1 {
 public:
  static constexpr double kPeriod = 4.5585e-3, kAzRate = 1567.45 * M_PI / 180.0, kTilt = 2.2 * M_PI / 180.0;
  static constexpr double kMinEl = 1.0 * M_PI / 180.0, kMaxRange = 30.0, kMinRange = 0.05, kHousing = 0.15;
  static constexpr double kNoise = 0.015;
  static constexpr long kSlots = 591, kCloudSlots = 8400;

  L1(const mjModel* m, const YAML::Node& lidar) : base_(id(m, mjOBJ_BODY, "base")) {
    std::array<double, 3> xyz{0.28945, 0.0, -0.046825}, rpy{0.0, 2.8782, 0.0};
    if (lidar && lidar["xyz"] && lidar["rpy"])
      for (int i = 0; i < 3; ++i) {
        xyz[i] = lidar["xyz"][i].as<double>();
        rpy[i] = lidar["rpy"][i].as<double>();
      }
    offset_ = get<double>(lidar, "range_offset", 0.0);
    const double cr = std::cos(rpy[0]), sr = std::sin(rpy[0]), cp = std::cos(rpy[1]), sp = std::sin(rpy[1]);
    const double cy = std::cos(rpy[2]), sy = std::sin(rpy[2]);
    const double r[9] = {cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, sy * cp, sy * sp * sr + cy * cr,
                         sy * sp * cr - cy * sr, -sp, cp * sr, cp * cr};
    std::copy(r, r + 9, rot_);
    std::copy(xyz.begin(), xyz.end(), pos_);
    pub_.reset(new ChannelPublisher<DdsCloud>("rt/utlidar/cloud"));
    pub_->InitChannel();
    imu_pub_.reset(new ChannelPublisher<DdsImu>("rt/utlidar/imu"));
    imu_pub_->InitChannel();
  }

  // Casts every slot up to the sim time; a finished cloud goes out stamped with its first slot's time.
  void step(const mjModel* m, const mjData* d) {
    const double* R = d->xmat + 9 * base_;
    double origin[3], rw[9];
    for (int i = 0; i < 3; ++i) {
      origin[i] = d->xpos[3 * base_ + i];
      for (int j = 0; j < 3; ++j) {
        origin[i] += R[3 * i + j] * pos_[j];
        rw[3 * i + j] = 0.0;
        for (int k = 0; k < 3; ++k) rw[3 * i + j] += R[3 * i + k] * rot_[3 * k + j];
      }
    }
    const long end = static_cast<long>(d->time * kSlots / kPeriod) + 1;
    for (; slot_ < end; ++slot_) {
      const double t = slot_ * kPeriod / kSlots, psi = 2.0 * M_PI * t / kPeriod, phi = kAzRate * t;
      const double c = std::cos(psi), s = std::sin(psi);
      const double v[3] = {c * std::cos(phi) - s * std::sin(kTilt) * std::sin(phi),
                           c * std::sin(phi) + s * std::sin(kTilt) * std::cos(phi), s * std::cos(kTilt)};
      if (v[2] >= std::sin(kMinEl)) {
        double w[3], start[3];
        for (int i = 0; i < 3; ++i) {
          w[i] = rw[3 * i] * v[0] + rw[3 * i + 1] * v[1] + rw[3 * i + 2] * v[2];
          start[i] = origin[i] + kMinRange * w[i];
        }
        int geom = -1;
        const double dist = mj_ray(m, d, start, w, kGroups, 1, -1, &geom, nullptr);
        if (dist >= 0.0 && dist < kMaxRange - kMinRange &&
            (dist >= kHousing - kMinRange || m->geom_bodyid[geom] != base_)) {
          const double r = dist + kMinRange + offset_ + noise_(rng_);
          const float p[4] = {float(v[0] * r), float(v[1] * r), float(v[2] * r),
                              float(t - (slot_ / kCloudSlots) * kCloudSlots * kPeriod / kSlots)};
          points_.insert(points_.end(), p, p + 4);
        }
      }
      if ((slot_ + 1) % kCloudSlots == 0) publish(d->time - (slot_ + 1 - kCloudSlots) * kPeriod / kSlots);
    }
    if (++imu_count_ % 2 == 0) {
      DdsImu imu;
      imu.header().stamp(wall_stamp(0.0));
      imu.header().frame_id("utlidar_imu");
      const double* g = d->qvel + 3;
      imu.angular_velocity().x(rot_[0] * g[0] + rot_[3] * g[1] + rot_[6] * g[2]);
      imu.angular_velocity().y(rot_[1] * g[0] + rot_[4] * g[1] + rot_[7] * g[2]);
      imu.angular_velocity().z(rot_[2] * g[0] + rot_[5] * g[1] + rot_[8] * g[2]);
      imu_pub_->Write(imu);
    }
  }

 private:
  using DdsCloud = sensor_msgs::msg::dds_::PointCloud2_;
  using DdsImu = sensor_msgs::msg::dds_::Imu_;
  static constexpr mjtByte kGroups[6] = {1, 0, 1, 0, 0, 0};

  void publish(double ago) {
    const size_t n = points_.size() / 4;
    DdsCloud msg;
    msg.header().stamp(wall_stamp(ago));
    msg.header().frame_id("utlidar_lidar");
    msg.height(1);
    msg.width(static_cast<uint32_t>(n));
    std::vector<sensor_msgs::msg::dds_::PointField_> fields;
    const std::pair<const char*, std::pair<uint32_t, uint8_t>> layout[] = {
        {"x", {0, 7}}, {"y", {4, 7}}, {"z", {8, 7}}, {"intensity", {16, 7}}, {"ring", {20, 4}}, {"time", {24, 7}}};
    for (const auto& [name, f] : layout) {
      sensor_msgs::msg::dds_::PointField_ pf;
      pf.name(name);
      pf.offset(f.first);
      pf.datatype(f.second);
      pf.count(1);
      fields.push_back(pf);
    }
    msg.fields(fields);
    msg.is_bigendian(false);
    msg.point_step(32);
    msg.row_step(static_cast<uint32_t>(32 * n));
    std::vector<uint8_t> data(32 * n, 0);
    const float intensity = 100.0f;
    for (size_t i = 0; i < n; ++i) {
      std::memcpy(&data[32 * i], &points_[4 * i], 12);
      std::memcpy(&data[32 * i + 16], &intensity, 4);
      std::memcpy(&data[32 * i + 24], &points_[4 * i + 3], 4);
    }
    msg.data(data);
    msg.is_dense(true);
    pub_->Write(msg);
    points_.clear();
  }

  int base_;
  double rot_[9], pos_[3], offset_ = 0.0;
  long slot_ = 0, imu_count_ = 0;
  std::vector<float> points_;
  std::mt19937 rng_{0};
  std::normal_distribution<double> noise_{0.0, kNoise};
  unitree::robot::ChannelPublisherPtr<DdsCloud> pub_;
  unitree::robot::ChannelPublisherPtr<DdsImu> imu_pub_;
};

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const std::string interface = arg(args, "--interface").empty() ? "lo" : arg(args, "--interface");
  const std::string scene = arg(args, "--scene").empty()
                                ? Paths::find().description + "/mujoco_menagerie/unitree_go2/scene.xml"
                                : arg(args, "--scene");
  const double duration = arg(args, "--duration").empty() ? 0.0 : std::stod(arg(args, "--duration"));
  const double max_torque = arg(args, "--max-torque").empty() ? 45.0 : std::stod(arg(args, "--max-torque"));

  mj_loadAllPluginLibraries(MUJOCO_PLUGIN_DIR, nullptr);  // the mesh decoders (.obj, .stl)
  char error[1000] = "";
  mjModel* m = mj_loadXML(scene.c_str(), nullptr, error, sizeof(error));
  if (!m) {
    std::cerr << "[mock_go2] " << scene << ": " << error << std::endl;
    return 1;
  }
  mjData* d = mj_makeData(m);
  int qadr[12], vadr[12], act[12], touch[4];
  for (int leg = 0; leg < 4; ++leg) {
    for (int part = 0; part < 3; ++part) {
      const std::string name = std::string(kLegs[leg]) + "_" + kParts[part];
      const int j = id(m, mjOBJ_JOINT, name + "_joint");
      qadr[leg * 3 + part] = m->jnt_qposadr[j];
      vadr[leg * 3 + part] = m->jnt_dofadr[j];
      act[leg * 3 + part] = id(m, mjOBJ_ACTUATOR, name);
    }
    touch[leg] = m->sensor_adr[id(m, mjOBJ_SENSOR, std::string(kLegs[leg]) + "_touch")];
  }
  const int accel = m->sensor_adr[id(m, mjOBJ_SENSOR, "accelerometer")];

  // Lying on its belly, as the robot is after Sport mode's StandDown.
  const double base[7] = {0.0, 0.0, 0.12, 1.0, 0.0, 0.0, 0.0};
  std::copy(base, base + 7, d->qpos);
  for (int i = 0; i < 12; ++i) d->qpos[qadr[i]] = kLie[i / 3][i % 3];
  mj_forward(m, d);

  unsetenv("CYCLONEDDS_URI");
  ChannelFactory::Instance()->Init(0, interface);
  ChannelPublisher<LowState> state_pub("rt/lowstate");
  state_pub.InitChannel();
  std::optional<L1> lidar;
  if (!flag(args, "--no_lidar")) lidar.emplace(m, load_config(Paths::find(), "mock_go2")["real_lidar"]);
  std::mutex lock;
  std::optional<LowCmd> latest;
  long count = 0, bad_crc = 0;
  ChannelSubscriber<LowCmd> cmd_sub("rt/lowcmd");
  cmd_sub.InitChannel(
      [&](const void* msg) {
        const auto& cmd = *static_cast<const LowCmd*>(msg);
        std::lock_guard<std::mutex> l(lock);
        ++count;
        if (low_cmd_crc(cmd) != cmd.crc()) {
          ++bad_crc;
          return;
        }
        latest = cmd;
      },
      10);

  LowState state{};
  const double dt = m->opt.timestep;
  const auto start = Clock::now();
  auto last_report = start;
  uint32_t step = 0;
  std::printf("[mock_go2] MuJoCo Go2 on '%s', %.0f Hz, waiting for rt/lowcmd.\n", interface.c_str(), 1.0 / dt);
  std::fflush(stdout);
  while (duration <= 0.0 || d->time < duration) {
    std::optional<LowCmd> cmd;
    {
      std::lock_guard<std::mutex> l(lock);
      cmd = latest;
    }
    for (int i = 0; i < 12; ++i) {
      double tau = 0.0;
      if (cmd) {
        const auto& mc = cmd->motor_cmd()[i];
        tau = mc.kp() * (mc.q() - d->qpos[qadr[i]]) + mc.kd() * (mc.dq() - d->qvel[vadr[i]]) + mc.tau();
      }
      d->ctrl[act[i]] = std::clamp(tau, -max_torque, max_torque);
    }
    mj_step(m, d);
    ++step;

    for (int i = 0; i < 12; ++i) {
      auto& ms = state.motor_state()[i];
      ms.q() = static_cast<float>(d->qpos[qadr[i]]);
      ms.dq() = static_cast<float>(d->qvel[vadr[i]]);
      ms.tau_est() = static_cast<float>(d->ctrl[act[i]]);
    }
    for (int k = 0; k < 4; ++k) state.imu_state().quaternion()[k] = static_cast<float>(d->qpos[3 + k]);
    for (int k = 0; k < 3; ++k) {
      state.imu_state().gyroscope()[k] = static_cast<float>(d->qvel[3 + k]);
      state.imu_state().accelerometer()[k] = static_cast<float>(d->sensordata[accel + k]);
    }
    for (int leg = 0; leg < 4; ++leg)
      state.foot_force()[leg] = static_cast<int16_t>(kFsrBias + d->sensordata[touch[leg]] / kFsrScale);
    state.tick() = step;
    state_pub.Write(state);
    if (lidar) lidar->step(m, d);

    const auto now = Clock::now();
    if (now - last_report >= std::chrono::seconds(2)) {
      last_report = now;
      std::lock_guard<std::mutex> l(lock);
      std::printf("[mock_go2] t=%6.1fs height %.2f m  vx %+.2f vy %+.2f  lowcmd %ld (%ld bad CRC)\n", d->time,
                  d->qpos[2], d->qvel[0], d->qvel[1], count, bad_crc);
      std::fflush(stdout);
    }
    std::this_thread::sleep_until(start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(d->time)));
  }
  cmd_sub.CloseChannel();
  mj_deleteData(d);
  mj_deleteModel(m);
  std::fflush(nullptr);
  std::_Exit(0);
}
