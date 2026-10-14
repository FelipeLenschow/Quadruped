// Joint tester for the hung-up robot: holds the pose it finds, moves every joint to the middle of
// [min_pct, max_pct] of its range, then swings one joint at a time (0.25 Hz sine over that span) on ENTER.
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

#include <rclcpp/utilities.hpp>

#include "config_loader.hpp"
#include "robot_io.hpp"

using namespace quadruped_real;
using Clock = std::chrono::steady_clock;

namespace {

// Go2 URDF limits, SDK order: FR, FL, RR, RL; hip, thigh, calf each.
const double kLimits[12][2] = {
    {-1.0472, 1.0472}, {-1.5708, 3.4907}, {-2.7227, -0.83776},  // FR
    {-1.0472, 1.0472}, {-1.5708, 3.4907}, {-2.7227, -0.83776},  // FL
    {-1.0472, 1.0472}, {-0.5236, 4.5379}, {-2.7227, -0.83776},  // RR
    {-1.0472, 1.0472}, {-0.5236, 4.5379}, {-2.7227, -0.83776},  // RL
};
const char* kNames[12] = {"FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
                          "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"};
constexpr double kInterpolationS = 5.0;
constexpr double kFrequency = 0.25;

enum State { kHoldInitial, kInterpolating, kHoldCenter, kTesting };

double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

bool enter(const std::string& prompt) {
  std::cout << prompt << std::flush;
  std::string line;
  return static_cast<bool>(std::getline(std::cin, line));
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const double min_pct = arg(args, "--min_pct").empty() ? 0.3 : std::stod(arg(args, "--min_pct"));
  const double max_pct = arg(args, "--max_pct").empty() ? 0.7 : std::stod(arg(args, "--max_pct"));

  std::cout << "\n" << std::string(50, '=') << "\n UNITREE JOINT TESTER \n" << std::string(50, '=') << "\n"
            << "WARNING: Ensure the robot is SUSPENDED safely! All joints will move to 50% of their range when started.\n"
            << std::string(50, '=') << std::endl;
  std::cout << "Initializing SDK and waiting for LowState to capture initial pose..." << std::endl;

  unsetenv("CYCLONEDDS_URI");
  RobotIO io;
  try {
    io.init(arg(args, "--interface"));
  } catch (const std::exception& e) {
    std::cerr << "[SDK2] Failed to initialize: " << e.what() << std::endl;
    return 1;
  }
  LowState state;
  uint64_t seq = 0;
  while (!io.take(state, seq)) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  double initial[12], center[12];
  for (int i = 0; i < 12; ++i) {
    initial[i] = state.motor_state()[i].q();
    center[i] = kLimits[i][0] + 0.5 * (min_pct + max_pct) * (kLimits[i][1] - kLimits[i][0]);
  }

  std::atomic<int> mode{kHoldInitial}, active{-1};
  std::atomic<bool> running{true};
  Clock::time_point interpolation_start, swing_start;
  std::thread loop([&] {
    LowCmd cmd = stopped_low_cmd();
    auto next = Clock::now();
    while (running) {
      const int m = mode, a = active;
      for (int i = 0; i < 12; ++i) {
        double q = initial[i];
        if (m == kInterpolating) {
          q = initial[i] + (center[i] - initial[i]) * std::min(seconds(interpolation_start) / kInterpolationS, 1.0);
        } else if (m == kHoldCenter || (m == kTesting && i != a)) {
          q = center[i];
        } else if (m == kTesting) {
          const double pct = 0.5 * (min_pct + max_pct) +
                             0.5 * (max_pct - min_pct) * std::sin(2.0 * M_PI * kFrequency * seconds(swing_start));
          q = kLimits[i][0] + pct * (kLimits[i][1] - kLimits[i][0]);
        }
        auto& mc = cmd.motor_cmd()[i];
        mc.q() = static_cast<float>(q);
        mc.dq() = 0.0f;
        mc.kp() = 45.0f;
        mc.kd() = 1.0f;
        mc.tau() = 0.0f;
      }
      io.write(cmd);
      next += std::chrono::milliseconds(2);  // 500 Hz
      std::this_thread::sleep_until(next);
    }
  });

  if (enter("\n[STEP 1] Initial pose captured! Press Enter to start 5s interpolation to center pose...")) {
    interpolation_start = Clock::now();
    mode = kInterpolating;
    std::cout << "Interpolating to center pose... (waiting 5 seconds)" << std::endl;
    std::this_thread::sleep_for(std::chrono::duration<double>(kInterpolationS));
    mode = kHoldCenter;
    if (enter("\n[STEP 2] Center pose reached! Press Enter to begin testing individual joints...")) {
      mode = kTesting;
      for (int i = 0; i < 12; ++i) {
        std::cout << "\n[" << i + 1 << "/12] Next joint to test: " << kNames[i] << std::endl;
        if (!enter("Press Enter to start oscillating...")) break;
        swing_start = Clock::now();
        active = i;
        const bool more = enter(std::string("--> Oscillating ") + kNames[i] + "! Press Enter to stop this joint...");
        active = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (!more) break;
      }
    }
  }
  running = false;
  loop.join();
  std::cout << "\nTest completed." << std::endl;
  io.close();
  std::fflush(nullptr);
  std::_Exit(0);
}
