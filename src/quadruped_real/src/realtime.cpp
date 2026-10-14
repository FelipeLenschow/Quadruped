#include "realtime.hpp"

#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>

namespace quadruped_real {

namespace {

std::set<int> online_cpus() {
  std::set<int> cpus;
  std::ifstream f("/sys/devices/system/cpu/online");
  std::string spec;
  std::getline(f, spec);
  std::stringstream ss(spec);
  std::string part;
  while (std::getline(ss, part, ',')) {
    const auto dash = part.find('-');
    const int lo = std::stoi(part.substr(0, dash));
    const int hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
    for (int c = lo; c <= hi; ++c) cpus.insert(c);
  }
  return cpus;
}

std::vector<int> numeric_entries(const std::string& dir) {
  std::vector<int> out;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      char* end = nullptr;
      const long v = std::strtol(e->d_name, &end, 10);
      if (end != e->d_name && *end == '\0') out.push_back(static_cast<int>(v));
    }
    closedir(d);
  }
  return out;
}

}  // namespace

std::optional<int> reserved_core(const RealtimeConfig& cfg) {
#if defined(__aarch64__)
  if (!cfg.enabled) return std::nullopt;
  const std::set<int> online = online_cpus();
  if (online.size() < 2) return std::nullopt;
  const int core = cfg.core < 0 ? *online.rbegin() : cfg.core;
  if (!online.count(core)) return std::nullopt;
  return core;
#else
  (void)cfg;
  return std::nullopt;
#endif
}

void move_other_threads_off(int core) {
  const int me = getpid();
  for (int pid : numeric_entries("/proc")) {
    if (pid == me) continue;
    for (int tid : numeric_entries("/proc/" + std::to_string(pid) + "/task")) {
      cpu_set_t set;
      if (sched_getaffinity(tid, sizeof(set), &set) != 0) continue;
      if (!CPU_ISSET(core, &set) || CPU_COUNT(&set) < 2) continue;
      CPU_CLR(core, &set);
      sched_setaffinity(tid, sizeof(set), &set);
    }
  }
}

void avoid_core(int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : online_cpus()) {
    if (c != core) CPU_SET(c, &set);
  }
  sched_setaffinity(0, sizeof(set), &set);
}

std::string take_core(int core, int priority) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) return std::string("could not pin to the core: ") + std::strerror(errno);
  sched_param param{};
  param.sched_priority = priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
    return "no permission for SCHED_FIFO (needs root, rtprio, or a container with a realtime budget); pinned only";
  }
  return "";
}

}  // namespace quadruped_real
