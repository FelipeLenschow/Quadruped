#include "quadruped_control/safety.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace quadruped_control {

namespace {

constexpr double kRadToDeg = 180.0 / M_PI;

std::string format(const char* fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

double f32(double v) { return static_cast<float>(v); }

}  // namespace

SafetyProcessor::SafetyProcessor(const ControllerConfig& cfg, EventQueue& events)
    : cfg_(cfg), events_(events), desired_qpos_(default_stance()), watchdog_timeout_(cfg.watchdog_timeout) {
  const float saturation = static_cast<float>(cfg.saturation);
  for (int i = 0; i < 12; ++i) {
    const float lo = static_cast<float>(cfg.hard_min[i]), hi = static_cast<float>(cfg.hard_max[i]);
    const float center = (lo + hi) / 2.0f, half = (hi - lo) / 2.0f;
    soft_min_[i] = center - half * saturation;
    soft_max_[i] = center + half * saturation;
  }
  limits_min_ = cfg.hard_min.unaryExpr([](double v) { return f32(v); });
  limits_max_ = cfg.hard_max.unaryExpr([](double v) { return f32(v); });
  recompute();
}

void SafetyProcessor::recompute() {
  if (rom_margin_pin_) rom_margin_ = *rom_margin_pin_;
  tilt_thresh_ = std::cos(base_tilt_limit_ / kRadToDeg);
  const Vec12 rom = limits_max_ - limits_min_;
  joint_safe_min_ = limits_min_ + rom_margin_ * rom;
  joint_safe_max_ = limits_max_ - rom_margin_ * rom;
  global_max_torque_ = (max_torque_percent_ / 100.0) * cfg_.motor_max_torque;
}

void SafetyProcessor::heartbeat(double now) {
  last_heartbeat_ = now;
  has_heartbeat_ = true;
}

void SafetyProcessor::set_max_torque_percent(double v) {
  max_torque_percent_ = v;
  recompute();
}

void SafetyProcessor::set_base_tilt_limit(double deg) {
  base_tilt_limit_ = deg;
  recompute();
}

void SafetyProcessor::set_forward_tilt_limit(double deg) {
  forward_tilt_limit_ = deg;
  recompute();
}

void SafetyProcessor::set_rom_margin(double margin) {
  rom_margin_ = margin;
  recompute();
}

void SafetyProcessor::set_watchdog_timeout(double seconds) {
  if (seconds > 0.0) watchdog_timeout_ = seconds;
}

bool SafetyProcessor::estop() {
  if (estop_) return false;
  estop_ = true;
  active_max_torque_ = 0.0;
  events_.log(LogLevel::Error, "[SAFETY] E-STOP latched via /safety/estop.");
  return true;
}

void SafetyProcessor::release_estop() {
  if (!estop_) return;
  estop_ = false;
  events_.log(LogLevel::Warn, "[SAFETY] E-STOP released at the robot.");
}

void SafetyProcessor::set_numb(const std::string& reason) {
  if (numb_) return;
  numb_ = true;
  numb_reason_ = reason;
  events_.log(LogLevel::Warn, "[SAFETY] Numb: " + reason + ". Release from the app or console.");
}

void SafetyProcessor::numb(bool on) {
  if (on) {
    set_numb("asked for");
  } else if (numb_) {
    release();
  }
}

void SafetyProcessor::release() {
  if (numb_) events_.log(LogLevel::Warn, "[SAFETY] Numb released.");
  numb_ = false;
  numb_reason_.clear();
  trip_reason_.clear();
  shutdown_logged_ = false;
  policy_blocked_ = false;
}

bool SafetyProcessor::safety_reset() {
  if (!(policy_blocked_ || numb_)) return false;
  events_.log(LogLevel::Warn, "[CommandSafetyProcessor] Safety reset received from console.");
  release();
  return true;
}

void SafetyProcessor::clear_block(bool clear_reason) {
  policy_blocked_ = false;
  shutdown_logged_ = false;
  if (clear_reason) trip_reason_.clear();
}

void SafetyProcessor::pin_rom_margin(std::optional<double> margin) {
  rom_margin_pin_ = margin;
  recompute();
}

bool SafetyProcessor::heartbeat_ok(double now) const {
  return has_heartbeat_ && (now - last_heartbeat_) <= watchdog_timeout_;
}

std::pair<bool, std::string> SafetyProcessor::evaluate(const State& state, uint8_t protections, double now) const {
  if (protections & kHeartbeat) {
    if (!has_heartbeat_) return {false, "No supervisor heartbeat received yet"};
    if (now - last_heartbeat_ > watchdog_timeout_) {
      return {false, format("Supervisor heartbeat lost (>%.1fs)", watchdog_timeout_)};
    }
  }

  const Vec3 g = projected_gravity_b(state.quat_wxyz);
  const double tilt_cos = -g[2] / 9.81;
  if ((protections & kTilt) && tilt_cos < tilt_thresh_) {
    const double tilt_deg = std::acos(std::clamp(tilt_cos, -1.0, 1.0)) * kRadToDeg;
    return {false, format("Base tilt %.1f° exceeds limit %.1f°", tilt_deg, base_tilt_limit_)};
  }

  const double pitch_deg = std::atan2(g[0], -g[2]) * kRadToDeg;
  if ((protections & kPitch) && pitch_deg > forward_tilt_limit_) {
    return {false, format("Forward pitch %.1f° exceeds limit %.1f°", pitch_deg, forward_tilt_limit_)};
  }

  if (protections & kRom) {
    for (int i = 0; i < 12; ++i) {
      const double v = state.q[i];
      const char* name = cfg_.joint_names[i].c_str();
      if (v < joint_safe_min_[i]) return {false, format("%s: %+.3f rad < safe min %.3f", name, v, joint_safe_min_[i])};
      if (v > joint_safe_max_[i]) return {false, format("%s: %+.3f rad > safe max %.3f", name, v, joint_safe_max_[i])};
    }
  }
  return {true, ""};
}

Vec12 SafetyProcessor::clip_soft(const Vec12& t) const { return t.cwiseMax(soft_min_).cwiseMin(soft_max_); }

std::pair<Vec12, double> SafetyProcessor::process(const std::optional<Vec12>& main, const std::optional<Vec12>& safety,
                                                  const State& state, uint8_t protections, double now) {
  // The e-stop outranks everything, the safety policy included.
  if (estop_ || numb_) {
    active_max_torque_ = 0.0;
    return {desired_qpos_, 0.0};
  }

  auto blocked = [&]() -> std::pair<Vec12, double> {
    if (safety) {
      active_max_torque_ = global_max_torque_;
      return {clip_soft(*safety), active_max_torque_};
    }
    active_max_torque_ = 0.0;
    return {desired_qpos_, 0.0};
  };

  if (policy_blocked_) return blocked();

  const auto [safe, reason] = evaluate(state, protections, now);
  if (safe) {
    active_max_torque_ = global_max_torque_;
    return {clip_soft(main.value_or(desired_qpos_)), active_max_torque_};
  }

  if (!shutdown_logged_) {
    trip_reason_ = reason;
    shutdown_logged_ = true;
    policy_blocked_ = true;
    events_.log(LogLevel::Error, "[SAFETY] EMERGENCY SHUTDOWN: " + reason);
    events_.push({EventKind::SafetyTrip, LogLevel::Error, reason});
  }
  return blocked();
}

}  // namespace quadruped_control
