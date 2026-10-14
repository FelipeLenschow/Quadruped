// The real Go2's driver. The control loop (LowState in, controller step, LowCmd out) runs alone on the
// reserved core at SCHED_FIFO with the SDK's DDS threads; ROS, policy loading and the operator's console run
// on the other cores and reach the loop only through its mailboxes.
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "control_loop.hpp"
#include "driver_node.hpp"
#include "realtime.hpp"

using namespace quadruped_real;

namespace {

template <typename T>
T get(const YAML::Node& n, const char* key, T fallback) {
  if (!n || !n.IsMap() || !n[key] || n[key].IsNull()) return fallback;
  return n[key].as<T>();
}

std::string arg(const std::vector<std::string>& args, const std::string& name) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].rfind(name + "=", 0) == 0) return args[i].substr(name.size() + 1);
    if (args[i] == name && i + 1 < args.size()) return args[i + 1];
  }
  return "";
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const std::string robot = arg(args, "--robot").empty() ? "go2" : arg(args, "--robot");
  const std::string checkpoint = arg(args, "--internal_policy");

  const Paths paths = Paths::find();
  YAML::Node config;
  try {
    config = YAML::LoadFile(paths.config);
  } catch (const std::exception& e) {
    std::cerr << "[RealDriver] Cannot read " << paths.config << ": " << e.what() << std::endl;
    return 1;
  }

  // The SDK talks on DDS domain 0; ROS gets its own domain, and a CYCLONEDDS_URI meant for ROS must not
  // reconfigure the SDK's participant.
  const char* domain_override = std::getenv("QUADRUPED_ROS_DOMAIN_ID");  // tests: keep off the stack's domain
  setenv("ROS_DOMAIN_ID",
         domain_override ? domain_override : get<std::string>(config["network"], "ros_domain_id", "1").c_str(), 1);
  unsetenv("CYCLONEDDS_URI");

  RealtimeConfig rt;
  rt.enabled = get<bool>(config["realtime"], "enabled", true);
  rt.core = get<int>(config["realtime"], "core", -1);
  rt.priority = get<int>(config["realtime"], "priority", 80);
  const std::optional<int> core = reserved_core(rt);
  if (core) {
    move_other_threads_off(*core);
    avoid_core(*core);  // this thread and everything it starts: ROS, loaders, the console
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
