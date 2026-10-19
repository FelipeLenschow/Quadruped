// A MuJoCo Go2 on the robot's own low-level DDS topics, to run the real driver against on a PC.
//
// rt/lowstate goes out at the physics rate (joints, IMU, foot force in raw FSR counts); rt/lowcmd comes in and
// is applied as the motors do it, tau = kp (q* - q) + kd (dq* - dq) + tau_ff, after the same CRC check the
// robot makes (a command with a bad CRC is ignored). Joints are in the SDK's order: FR, FL, RR, RL,
// hip/thigh/calf each. Use the loopback interface on both sides (--interface lo here and on the driver) so
// nothing reaches a real robot. A MuJoCo window shows the robot (where GLFW was found at build time); --headless
// runs without it. Closing the window stops the mock.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#ifdef MOCK_GO2_VIEWER
#include <GLFW/glfw3.h>
#endif
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

#ifdef MOCK_GO2_VIEWER
// The window. The camera follows the base: drag with the left button to turn it, scroll to zoom.
class Viewer {
 public:
  explicit Viewer(const mjModel* m) : m_(m) {
    glfwSetErrorCallback([](int, const char* text) { std::fprintf(stderr, "[mock_go2] GLFW: %s\n", text); });
    if (!glfwInit()) return;
    window_ = glfwCreateWindow(1200, 900, "mock_go2", nullptr, nullptr);
    if (!window_) {
      glfwTerminate();
      return;
    }
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);
    mjv_makeScene(m, &scn_, 2000);
    mjr_makeContext(m, &con_, mjFONTSCALE_150);
    const int base = mj_name2id(m, mjOBJ_BODY, "base");
    cam_.type = mjCAMERA_TRACKING;
    cam_.trackbodyid = base < 0 ? 1 : base;
    cam_.distance = 1.5;
    cam_.azimuth = 135.0;
    cam_.elevation = -20.0;
    glfwSetWindowUserPointer(window_, this);
    glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) {
      static_cast<Viewer*>(glfwGetWindowUserPointer(w))->move(x, y);
    });
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double, double dy) {
      auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
      mjv_moveCamera(v->m_, mjMOUSE_ZOOM, 0.0, -0.05 * dy, &v->cam_);
    });
  }
  ~Viewer() {
    if (!window_) return;
    mjv_freeScene(&scn_);
    mjr_freeContext(&con_);
    glfwDestroyWindow(window_);
    glfwTerminate();
  }
  bool open() const { return window_ && !glfwWindowShouldClose(window_); }

  // Copy the state into the scene under the simulation's lock, draw without it.
  void render(mjData* d, std::mutex& sim_lock) {
    {
      std::lock_guard<std::mutex> l(sim_lock);
      mjv_updateScene(m_, d, &opt_, nullptr, &cam_, mjCAT_ALL, &scn_);
    }
    mjrRect viewport{0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
    mjr_render(viewport, &scn_, &con_);
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }

 private:
  void move(double x, double y) {
    const double dx = x - x_, dy = y - y_;
    x_ = x;
    y_ = y;
    if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS) return;
    int width, height;
    glfwGetWindowSize(window_, &width, &height);
    mjv_moveCamera(m_, mjMOUSE_ROTATE_V, dx / height, dy / height, &cam_);
  }

  const mjModel* m_;
  GLFWwindow* window_ = nullptr;
  mjvCamera cam_;
  mjvOption opt_;
  mjvScene scn_;
  mjrContext con_;
  double x_ = 0.0, y_ = 0.0;
};
#endif

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const std::string interface = arg(args, "--interface").empty() ? "lo" : arg(args, "--interface");
  const std::string scene = arg(args, "--scene").empty()
                                ? Paths::find().description + "/mujoco_menagerie/unitree_go2/scene.xml"
                                : arg(args, "--scene");
  const double duration = arg(args, "--duration").empty() ? 0.0 : std::stod(arg(args, "--duration"));
  const double max_torque = arg(args, "--max-torque").empty() ? 45.0 : std::stod(arg(args, "--max-torque"));
  const bool headless = flag(args, "--headless");

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

  std::mutex sim_lock;  // d, between the physics and the window
  std::atomic<bool> stop{false};
  const double dt = m->opt.timestep;
  std::printf("[mock_go2] MuJoCo Go2 on '%s', %.0f Hz, waiting for rt/lowcmd.\n", interface.c_str(), 1.0 / dt);
  std::fflush(stdout);
  auto physics = [&] {
    LowState state{};
    const auto start = Clock::now();
    auto last_report = start;
    uint32_t step = 0;
    while (!stop) {
      std::unique_lock<std::mutex> sim(sim_lock);
      if (duration > 0.0 && d->time >= duration) break;
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

      const auto now = Clock::now();
      if (now - last_report >= std::chrono::seconds(2)) {
        last_report = now;
        std::lock_guard<std::mutex> l(lock);
        std::printf("[mock_go2] t=%6.1fs height %.2f m  vx %+.2f vy %+.2f  lowcmd %ld (%ld bad CRC)\n", d->time,
                    d->qpos[2], d->qvel[0], d->qvel[1], count, bad_crc);
        std::fflush(stdout);
      }
      const auto wake = start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(d->time));
      sim.unlock();
      state_pub.Write(state);
      std::this_thread::sleep_until(wake);
    }
    stop = true;
  };

#ifdef MOCK_GO2_VIEWER
  // GLFW runs on the main thread: the physics goes to a thread of its own.
  std::unique_ptr<Viewer> viewer;
  if (!headless) {
    viewer = std::make_unique<Viewer>(m);
    if (!viewer->open()) {
      std::fprintf(stderr, "[mock_go2] No window (is DISPLAY set?): headless.\n");
      viewer.reset();
    }
  }
  if (viewer) {
    std::thread sim_thread(physics);
    while (!stop && viewer->open()) viewer->render(d, sim_lock);
    stop = true;
    sim_thread.join();
    viewer.reset();
  } else {
    physics();
  }
#else
  if (!headless) std::fprintf(stderr, "[mock_go2] Built without GLFW: headless.\n");
  physics();
#endif
  cmd_sub.CloseChannel();
  mj_deleteData(d);
  mj_deleteModel(m);
  std::fflush(nullptr);
  std::_Exit(0);
}
