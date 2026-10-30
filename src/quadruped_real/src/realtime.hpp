#pragma once

#include <optional>
#include <string>

namespace quadruped_real {

// config.yaml's realtime block (quadruped_common/realtime.py).
struct RealtimeConfig {
  bool enabled = true;
  int core = -1;  // -1: the last online CPU
  int priority = 80;
};

// The control loop's core: on the robot (aarch64) with more than one CPU and realtime enabled.
std::optional<int> reserved_core(const RealtimeConfig& cfg);
// Every other thread on the machine that may run on `core` and on others too: off it.
void move_other_threads_off(int core);
// The calling thread (and what it starts later): every online CPU but `core`.
void avoid_core(int core);
// The calling thread: alone on `core` at SCHED_FIFO. Returns an error text when FIFO was refused (still pinned).
std::string take_core(int core, int priority);
// The whole process: every page it touches stays in RAM, so other processes' memory use (detector, LLM) cannot
// push the loop into swap and stall a step. Returns an error text when the lock was refused.
std::string lock_memory();

}  // namespace quadruped_real
