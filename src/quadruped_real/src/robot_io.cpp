#include "robot_io.hpp"

#include <unistd.h>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/go2/vui/vui_client.hpp>

namespace quadruped_real {

using namespace unitree::robot;

// VuiClient plus the light color call the Go2 app uses (api 1007), which the SDK leaves out.
class LightClient : public go2::VuiClient {
 public:
  void Init() {
    go2::VuiClient::Init();
    RegistApi(kSetColor, 0);
  }
  int32_t SetColor(const std::string& color, int seconds) {
    std::string reply;
    return Call(kSetColor, "{\"color\":\"" + color + "\",\"time\":" + std::to_string(seconds) + "}", reply);
  }

 private:
  static constexpr int32_t kSetColor = 1007;
};

RobotIO::RobotIO() = default;
RobotIO::~RobotIO() = default;

void RobotIO::init(const std::string& interface) {
  ChannelFactory::Instance()->Init(0, interface);
  publisher_.reset(new ChannelPublisher<LowCmd>("rt/lowcmd"));
  publisher_->InitChannel();
  subscriber_.reset(new ChannelSubscriber<LowState>("rt/lowstate"));
  subscriber_->InitChannel([this](const void* msg) { on_low_state(msg); }, 1);
  vui_ = std::make_unique<LightClient>();
  vui_->SetTimeout(5.0f);
  vui_->Init();
}

void RobotIO::release_motion_mode(const std::function<void(const std::string&)>& log) {
  go2::SportClient sport;
  sport.SetTimeout(5.0f);
  sport.Init();
  b2::MotionSwitcherClient switcher;
  switcher.SetTimeout(5.0f);
  switcher.Init();
  std::string form, name;
  int32_t ret = switcher.CheckMode(form, name);
  if (ret != 0) log("[RealDriver] CheckMode failed (error " + std::to_string(ret) + "); assuming no motion mode is active.");
  while (ret == 0 && !name.empty()) {
    log("[RealDriver] Robot is in mode '" + name + "' - calling StandDown + ReleaseMode.");
    sport.StandDown();
    switcher.ReleaseMode();
    ret = switcher.CheckMode(form, name);
    sleep(1);
  }
}

void RobotIO::on_low_state(const void* msg) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  state_ = *static_cast<const LowState*>(msg);
  ++state_seq_;
}

bool RobotIO::take(LowState& out, uint64_t& seq) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (state_seq_ == seq) return false;
  out = state_;
  seq = state_seq_;
  return true;
}

void RobotIO::write(LowCmd& cmd) {
  cmd.crc() = low_cmd_crc(cmd);
  publisher_->Write(cmd);
}

int32_t RobotIO::set_color(const std::string& color, int seconds) { return vui_ ? vui_->SetColor(color, seconds) : -1; }

void RobotIO::close() {
  if (subscriber_) subscriber_->CloseChannel();
  subscriber_.reset();
  publisher_.reset();
  vui_.reset();
}

LowCmd stopped_low_cmd() {
  LowCmd cmd{};
  cmd.head()[0] = 0xFE;
  cmd.head()[1] = 0xEF;
  cmd.level_flag() = 0xFF;
  cmd.gpio() = 0;
  for (int i = 0; i < 20; ++i) {
    cmd.motor_cmd()[i].mode() = 0x01;
    cmd.motor_cmd()[i].q() = 2.146e9f;  // PosStopF
    cmd.motor_cmd()[i].kp() = 0.0f;
    cmd.motor_cmd()[i].dq() = 16000.0f;  // VelStopF
    cmd.motor_cmd()[i].kd() = 0.0f;
    cmd.motor_cmd()[i].tau() = 0.0f;
  }
  return cmd;
}

// Unitree's CRC32 over the command's 32-bit words, the crc field itself excluded.
uint32_t low_cmd_crc(const LowCmd& cmd) {
  const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&cmd);
  const uint32_t len = (sizeof(LowCmd) >> 2) - 1;
  uint32_t crc = 0xFFFFFFFF;
  const uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t bit = 1u << 31;
    const uint32_t data = ptr[i];
    for (int b = 0; b < 32; ++b) {
      crc = (crc & 0x80000000) ? (crc << 1) ^ poly : crc << 1;
      if (data & bit) crc ^= poly;
      bit >>= 1;
    }
  }
  return crc;
}

}  // namespace quadruped_real
