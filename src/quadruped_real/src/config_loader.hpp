#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "paths.hpp"

namespace quadruped_real {

// config.yaml (quadruped_common/config_loader.py); an empty node if it cannot be read.
inline YAML::Node load_config(const Paths& paths, const char* who) {
  try {
    return YAML::LoadFile(paths.config);
  } catch (const std::exception& e) {
    std::cerr << "[" << who << "] Cannot read " << paths.config << ": " << e.what() << std::endl;
    return YAML::Node();
  }
}

template <typename T>
T get(const YAML::Node& n, const char* key, T fallback) {
  if (!n || !n.IsMap() || !n[key] || n[key].IsNull()) return fallback;
  return n[key].as<T>();
}

// --name=value or --name value from the arguments rclcpp left; "" if absent.
inline std::string arg(const std::vector<std::string>& args, const std::string& name) {
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i].rfind(name + "=", 0) == 0) return args[i].substr(name.size() + 1);
    if (args[i] == name && i + 1 < args.size()) return args[i + 1];
  }
  return "";
}

inline bool flag(const std::vector<std::string>& args, const std::string& name) {
  for (size_t i = 1; i < args.size(); ++i)
    if (args[i] == name) return true;
  return false;
}

// The SDK talks on DDS domain 0; ROS gets its own domain (QUADRUPED_ROS_DOMAIN_ID overrides config.yaml's, for
// tests that must stay off a running stack), and a CYCLONEDDS_URI meant for ROS must not reconfigure the SDK.
inline void use_ros_domain(const YAML::Node& config) {
  const char* domain_override = std::getenv("QUADRUPED_ROS_DOMAIN_ID");
  setenv("ROS_DOMAIN_ID",
         domain_override ? domain_override : get<std::string>(config["network"], "ros_domain_id", "1").c_str(), 1);
  unsetenv("CYCLONEDDS_URI");
}

}  // namespace quadruped_real
