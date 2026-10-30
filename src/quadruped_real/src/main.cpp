// The real Go2's driver. The control loop (LowState in, controller step, LowCmd out) runs alone on the
// reserved core at SCHED_FIFO with the SDK's DDS threads; ROS, policy loading and the operator's console run
// on the other cores and reach the loop only through its mailboxes.
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "config_loader.hpp"
#include "control_loop.hpp"
#include "driver_node.hpp"
#include "realtime.hpp"

using namespace quadruped_real;

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const std::string robot = arg(args, "--robot").empty() ? "go2" : arg(args, "--robot");
  const std::string checkpoint = arg(args, "--internal_policy");

  const Paths paths = Paths::find();
  const YAML::Node config = load_config(paths, "RealDriver");
  if (!config.IsMap()) return 1;
  use_ros_domain(config);

  RealtimeConfig rt;
  rt.enabled = get<bool>(config["realtime"], "enabled", true);
  rt.core = get<int>(config["realtime"], "core", -1);
  rt.priority = get<int>(config["realtime"], "priority", 80);
  const std::optional<int> core = reserved_core(rt);
  std::string lock_err;
  if (core) {
    move_other_threads_off(*core);
    avoid_core(*core);  // this thread and everything it starts: ROS, loaders, the console
    lock_err = lock_memory();
  }

  LoopConfig lc;
  lc.interface = arg(args, "--interface");
  lc.core = core;
  lc.priority = rt.priority;
  lc.kp = get<double>(config["control"], "kp", 0.0);
  lc.kd = get<double>(config["control"], "kd", 0.0);
  lc.emergency_kd = get<double>(config["safety"], "emergency_kd", 5.0);
  lc.emergency_kd_rear = get<double>(config["safety"], "emergency_kd_rear", lc.emergency_kd);
  const YAML::Node est = config["state_estimator"];
  lc.contact_threshold = get<double>(est, "contact_threshold", 10.0);
  lc.fsr_bias = get<double>(est, "fsr_bias", 16.0);
  lc.calib_time = get<double>(est, "fsr_calibration_time", 1.0);
  lc.calib_offset_max = get<double>(est, "fsr_offset_max", 25.0);
  lc.calib_spread_max = get<double>(est, "fsr_spread_max", 3.0);

  CommandBox commands;
  SnapshotBox snapshots;
  ControlLoop loop(lc, commands, snapshots);
  std::string note;
  try {
    note = loop.start().get();
  } catch (const std::exception& e) {
    std::cerr << "[SDK2] Failed to initialize: " << e.what() << std::endl;
    loop.stop();
    return 1;
  }

  rclcpp::init(argc, argv);
  auto node = std::make_shared<DriverNode>(paths, config, loop, commands, snapshots);
  if (!note.empty()) RCLCPP_INFO(node->get_logger(), "%s", note.c_str());
  if (core) {
    if (lock_err.empty()) RCLCPP_INFO(node->get_logger(), "[RealDriver] Memory locked in RAM: swapping cannot stall the loop.");
    else RCLCPP_WARN(node->get_logger(), "[RealDriver] %s", lock_err.c_str());
  }
  RCLCPP_INFO(node->get_logger(), "[RealDriver] Contact threshold: %.0f counts above each foot's offset (start %.0f, %s).",
              lc.contact_threshold, lc.fsr_bias,
              lc.calib_time > 0 ? ("calibrating over " + std::to_string(lc.calib_time).substr(0, 3) + "s").c_str()
                                : "calibration disabled");
  if (lc.calib_time > 0) {
    RCLCPP_WARN(node->get_logger(), "[RealDriver] Keep the feet UNLOADED for the next %.1fs - the FSR zero is being measured now.",
                lc.calib_time);
  }
  loop.run(node->make_controller(checkpoint));
  RCLCPP_INFO(node->get_logger(), "[RealDriver] Initialized for %s (C++ driver). Ready for deployment.", robot.c_str());

  rclcpp::spin(node);
  loop.stop();
  node.reset();
  loop.close();
  if (rclcpp::ok()) rclcpp::shutdown();
  // A policy loader may still be running; it is detached and owns nothing the exit needs.
  std::fflush(nullptr);
  std::_Exit(0);
}
