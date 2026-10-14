// A MuJoCo Go2 on the robot's own low-level DDS topics, to run the real driver against on a PC.
//
// rt/lowstate goes out at the physics rate (joints, IMU, foot force in raw FSR counts); rt/lowcmd comes in and
// is applied as the motors do it, tau = kp (q* - q) + kd (dq* - dq) + tau_ff, after the same CRC check the
// robot makes (a command with a bad CRC is ignored). Joints are in the SDK's order: FR, FL, RR, RL,
// hip/thigh/calf each. Use the loopback interface on both sides (--interface lo here and on the driver) so
// nothing reaches a real robot.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include <rclcpp/utilities.hpp>

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
