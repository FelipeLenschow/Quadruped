#pragma once

#include <string>

namespace quadruped_real {

// Where the repo's data lives, from the colcon install or straight from src/ (quadruped_common/paths.py).
struct Paths {
  std::string config, policies, geometry, description, repo;
  static Paths find();
  std::string resolve(const std::string& path) const;  // policies.json paths are relative to the repo
};

std::string realpath_or(const std::string& path);

}  // namespace quadruped_real
