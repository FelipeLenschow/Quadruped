#include "paths.hpp"

#include <climits>
#include <cstdlib>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace quadruped_real {

namespace {

std::string dirname(const std::string& p) {
  const auto slash = p.find_last_of('/');
  return slash == std::string::npos ? "." : p.substr(0, slash);
}

}  // namespace

std::string realpath_or(const std::string& p) {
  char buf[PATH_MAX];
  return ::realpath(p.c_str(), buf) ? std::string(buf) : p;
}

Paths Paths::find() {
  Paths p;
  std::string bringup, description;
  try {
    bringup = ament_index_cpp::get_package_share_directory("quadruped_bringup");
    description = ament_index_cpp::get_package_share_directory("quadruped_description");
  } catch (const std::exception&) {
  }
  const char* repo_env = std::getenv("QUADRUPED_REPO");
  if (bringup.empty() || description.empty()) {
    const std::string repo = repo_env ? repo_env : ".";
    bringup = repo + "/src/quadruped_bringup";
    description = repo + "/src/quadruped_description";
  }
  p.config = bringup + "/config/config.yaml";
  p.policies = bringup + "/config/policies.json";
  p.geometry = description + "/config/go2_kinematics.yaml";
  p.description = description;
  // --symlink-install links config/ back into src/quadruped_bringup, which sits in the repo.
  p.repo = repo_env ? repo_env : dirname(dirname(dirname(dirname(realpath_or(p.config)))));
  return p;
}

std::string Paths::resolve(const std::string& path) const {
  if (path.empty() || path[0] == '/') return path;
  return repo + "/" + path;
}

}  // namespace quadruped_real
