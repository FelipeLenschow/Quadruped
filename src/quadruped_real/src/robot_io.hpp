#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace quadruped_real {

class LightClient;

using LowState = unitree_go::msg::dds_::LowState_;
using LowCmd = unitree_go::msg::dds_::LowCmd_;

// The Go2's low-level interface through Unitree SDK2: rt/lowstate in, rt/lowcmd out.
class RobotIO {
 public:
  RobotIO();
  ~RobotIO();

  // DDS participant, LowState reader, LowCmd writer. The DDS threads start here and inherit the
  // calling thread's core and scheduling, so call it from the control loop's thread.
  void init(const std::string& interface);
  // Sport mode off (StandDown + ReleaseMode until CheckMode names nothing), so low-level commands take effect.
  void release_motion_mode(const std::function<void(const std::string&)>& log);
  // Copies the newest LowState if one arrived after `seq`, and advances `seq`.
  bool take(LowState& out, uint64_t& seq);
  void write(LowCmd& cmd);
  // The light's color ("blue", "green", "red", ...) for `seconds`, then the robot's own again. 0 when accepted.
  int32_t set_color(const std::string& color, int seconds);
  // Drops the DDS endpoints and clients while the SDK is still whole.
  void close();

 private:
  void on_low_state(const void* msg);

  unitree::robot::ChannelPublisherPtr<LowCmd> publisher_;
  unitree::robot::ChannelSubscriberPtr<LowState> subscriber_;
  std::unique_ptr<LightClient> vui_;
  std::mutex state_mutex_;
  LowState state_{};
  uint64_t state_seq_ = 0;
};

// LowCmd with every motor stopped (PosStopF, VelStopF, zero gains): what the robot gets before any targets.
LowCmd stopped_low_cmd();
uint32_t low_cmd_crc(const LowCmd& cmd);

}  // namespace quadruped_real
