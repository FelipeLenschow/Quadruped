#include "control_loop.hpp"

#include <time.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace quadruped_real {

namespace {

// Joint order: the SDK's motors are FR, FL, RR, RL (hip, thigh, calf each); the controller's is Isaac's.
constexpr int kSdkToRos[12] = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};
constexpr int kRosToSdk[12] = {1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10};

void add_ns(timespec& t, long ns) {
  t.tv_nsec += ns;
  while (t.tv_nsec >= 1000000000L) {
    t.tv_nsec -= 1000000000L;
    ++t.tv_sec;
  }
}

bool later(const timespec& a, const timespec& b) {
  return a.tv_sec > b.tv_sec || (a.tv_sec == b.tv_sec && a.tv_nsec > b.tv_nsec);
}

}  // namespace

double steady_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double wall_now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

ControlLoop::ControlLoop(LoopConfig cfg, CommandBox& commands, SnapshotBox& snapshots)
    : cfg_(std::move(cfg)), commands_(commands), snapshots_(snapshots), low_cmd_(stopped_low_cmd()) {
  fsr_offset_ = Eigen::Vector4d::Constant(cfg_.fsr_bias);
  calib_n_ = std::max(0, static_cast<int>(std::lround(cfg_.calib_time / kPolicyDt)));
  calib_done_ = calib_n_ == 0;
}

ControlLoop::~ControlLoop() { stop(); }

std::future<std::string> ControlLoop::start() {
  std::promise<std::string> ready;
  auto future = ready.get_future();
  thread_ = std::thread(&ControlLoop::thread_main, this, std::move(ready));
  return future;
}

void ControlLoop::run(std::unique_ptr<qc::Controller> controller) { controller_promise_.set_value(std::move(controller)); }

void ControlLoop::stop() {
  running_ = false;
  try {
    controller_promise_.set_value(nullptr);
  } catch (const std::future_error&) {
  }
  if (thread_.joinable()) thread_.join();
}

void ControlLoop::close() { robot_.close(); }

void ControlLoop::thread_main(std::promise<std::string> ready) {
  std::string note;
  if (cfg_.core) {
    const std::string err = take_core(*cfg_.core, cfg_.priority);
    note = "[RealDriver] Policy core: " + std::to_string(*cfg_.core) +
           (err.empty() ? " (SCHED_FIFO " + std::to_string(cfg_.priority) + "), everything else moved off it" : " - " + err);
  }
  try {
    robot_.init(cfg_.interface);
    robot_.release_motion_mode([](const std::string& line) { std::cerr << line << std::endl; });
  } catch (...) {
    ready.set_exception(std::current_exception());
    return;
  }
  ready.set_value(note);

  controller_ = controller_promise_.get_future().get();
  if (controller_ && running_) loop();
}

void ControlLoop::loop() {
  timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);
  long tick = 0;
  while (running_) {
    add_ns(next, static_cast<long>(kLowCmdDt * 1e9));
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
    timespec woke;
    clock_gettime(CLOCK_MONOTONIC, &woke);
    late_max_ = std::max(late_max_, (woke.tv_sec - next.tv_sec) + (woke.tv_nsec - next.tv_nsec) * 1e-9);
    if (tick % kTicksPerPolicy == 0) {
      const double t0 = steady_now();
      try {
        policy_tick();
      } catch (const std::exception& e) {
        fault(e.what());
      }
      const double took = steady_now() - t0;
      step_sum_ += took;
      step_max_ = std::max(step_max_, took);
      ++step_count_;
    } else if (last_targets_) {
      // The motors run the PD on the last q/kp/kd they got; resending keeps the stream fresh and lets an
      // e-stop between policy steps reach them within one tick.
      send(*last_targets_);
    }
    ++tick;
    // Behind by more than a tick: drop the missed ones instead of bursting to catch up.
    timespec now, limit = next;
    clock_gettime(CLOCK_MONOTONIC, &now);
    add_ns(limit, static_cast<long>(kLowCmdDt * 1e9));
    if (later(now, limit)) {
      skipped_ += static_cast<long>(((now.tv_sec - next.tv_sec) + (now.tv_nsec - next.tv_nsec) * 1e-9) / kLowCmdDt);
      next = now;
    }
  }
}

void ControlLoop::fault(const std::string& what) {
  // Hold the last targets with zero stiffness and the e-stop damping: the robot sinks under control.
  controller_->safety().set_numb(std::string("control step failed: ") + what);
  auto snap = std::make_unique<Snapshot>();
  log(qc::LogLevel::Error, "[RealDriver] Control step failed: " + what + " - damping only.", *snap);
  snapshots_.push(std::move(snap));
  if (last_targets_) send(*last_targets_);
}

void ControlLoop::log(qc::LogLevel level, std::string text, Snapshot& snap) {
  snap.events.push_back({qc::EventKind::Log, level, std::move(text)});
}

void ControlLoop::policy_tick() {
  auto snap = std::make_unique<Snapshot>();
  const double now = steady_now();
  snap->sim_time = now;
  snap->wall_time = wall_now();

  Command c;
  while (commands_.pop(c)) apply(c, *snap);

  qc::Controller& ctl = *controller_;
  if (read_low_state(now, *snap)) {
    const qc::RawState raw = raw_state(*snap);
    check_health(*snap);
    const qc::StepResult res = ctl.step(raw, std::nullopt, now, now);
    // Telemetry only (nothing to the motors) until a main policy is loaded.
    if (res.policy_step && ctl.has_policy("main")) {
      last_targets_ = res.targets.cast<float>();
      send(*last_targets_);
    }
    snap->has_state = true;
    snap->state = ctl.state();
    snap->targets = res.targets;
    snap->max_torque = res.max_torque;
    snap->pose_status = res.pose_status;
  }

  const qc::SafetyProcessor& sp = ctl.safety();
  snap->mode = ctl.mode();
  snap->transition = ctl.transition_active();
  snap->pose_name = ctl.pose().current_pose_name();
  snap->pose_done = ctl.pose().is_complete();
  snap->estop = sp.is_estopped();
  snap->numb = sp.is_numb();
  snap->blocked = sp.is_policy_blocked();
  snap->safety_policy_active = ctl.safety_policy_active();
  snap->trip_reason = sp.trip_reason();
  snap->numb_reason = sp.numb_reason();
  snap->global_max_torque = sp.global_max_torque();
  if (!snap->has_state) snap->max_torque = sp.active_max_torque();
  snap->active = ctl.active_policy();
  snap->pending = ctl.pending_policy();
  snap->error = ctl.policy_error();
  snap->available = ctl.available();
  if (snap->active) {
    const qc::PolicyEntry e = ctl.entry(*snap->active);
    snap->protections = qc::protection_names(e.protections);
    snap->trick_switches = e.switches;
  }
  snap->trick = ctl.trick_command(now);
  if (const Eigen::VectorXf* g = ctl.gait_command()) snap->gait = *g;
  snap->gait_override = ctl.gait_override_active();
  if (ctl.last_command().size() >= 3) snap->command = ctl.last_command().head<3>();
  snap->fsr_offset = fsr_offset_;
  snap->contact_threshold = cfg_.contact_threshold;
  for (auto& e : ctl.drain_events()) snap->events.push_back(std::move(e));
  if (stats_start_ == 0.0) stats_start_ = now;
  if (now - stats_start_ >= 10.0 && step_count_ > 0) {
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "[RealDriver] Loop, last %.0f s: policy step %.2f ms mean / %.2f ms max, woke up to %.2f ms late, "
                  "%ld ticks skipped.",
                  now - stats_start_, 1e3 * step_sum_ / step_count_, 1e3 * step_max_, 1e3 * late_max_, skipped_);
    log(qc::LogLevel::Info, buf, *snap);
    stats_start_ = now;
    step_sum_ = step_max_ = late_max_ = 0.0;
    step_count_ = skipped_ = 0;
  }
  snapshots_.push(std::move(snap));
}

void ControlLoop::apply(Command& c, Snapshot& snap) {
  qc::Controller& ctl = *controller_;
  qc::SafetyProcessor& sp = ctl.safety();
  using K = Command::Kind;
  switch (c.kind) {
    case K::None: break;
    case K::Mode: ctl.set_mode(c.text); break;
    case K::PoseCommand: ctl.pose_command(c.text); break;
    case K::PoseDuration: ctl.set_pose_interp_duration(c.value); break;
    case K::CmdVel: ctl.cmd_vel(c.values.at(0), c.values.at(1), c.values.at(2), c.now); break;
    case K::Trick: ctl.trick(c.values, c.now); break;
    case K::Gait: ctl.gait(Eigen::Map<const Eigen::VectorXd>(c.values.data(), c.values.size()).cast<float>()); break;
    case K::GaitOverride:
      ctl.gait_override(Eigen::Map<const Eigen::VectorXd>(c.values.data(), c.values.size()).cast<float>(), c.now);
      break;
    case K::Heartbeat: sp.heartbeat(c.now); break;
    case K::MaxTorque: sp.set_max_torque_percent(c.value); break;
    case K::TiltLimit: sp.set_base_tilt_limit(c.value); break;
    case K::ForwardTiltLimit: sp.set_forward_tilt_limit(c.value); break;
    case K::RomMargin: sp.set_rom_margin(c.value); break;
    case K::WatchdogTimeout: sp.set_watchdog_timeout(c.value); break;
    case K::SafetyReset: sp.safety_reset(); break;
    case K::Estop: sp.estop(); break;
    case K::Numb: sp.numb(c.flag); break;
    case K::ReleaseEstop: sp.release_estop(); break;
    case K::ReleaseTrip: sp.release(); break;
    case K::RequestPolicy:
      if (ctl.request_policy(c.text) == qc::PolicyRequest::NeedsLoad) snap.load_requests.push_back(c.text);
      break;
    case K::AddPolicy: ctl.add_policy(c.text, std::move(*c.runner), c.entry); break;
    case K::LoadError: ctl.set_load_error(c.text, c.detail); break;
    case K::Scan: scan_ = Eigen::Map<const Eigen::VectorXd>(c.values.data(), c.values.size()).cast<float>(); break;
    case K::Kp:
      if (c.value != cfg_.kp) {
        cfg_.kp = c.value;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "[RealDriver] Dynamic Kp updated to: %.1f", c.value);
        log(qc::LogLevel::Info, buf, snap);
      }
      break;
    case K::Kd:
      if (c.value != cfg_.kd) {
        cfg_.kd = c.value;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "[RealDriver] Dynamic Kd updated to: %.2f", c.value);
        log(qc::LogLevel::Info, buf, snap);
      }
      break;
  }
}

bool ControlLoop::read_low_state(double now, Snapshot& snap) {
  if (robot_.take(low_state_, low_state_seq_)) {
    have_low_state_ = true;
    low_state_time_ = now;
    if (low_state_stale_) {
      low_state_stale_ = false;
      log(qc::LogLevel::Info, "[RealDriver] LowState stream resumed.", snap);
    }
  } else if (have_low_state_ && !low_state_stale_ && now - low_state_time_ > kLowStateStale) {
    low_state_stale_ = true;
    log(qc::LogLevel::Error,
        "[RealDriver] No LowState for 100 ms - the policy is running on the last sample received.", snap);
  }
  return have_low_state_;
}

void ControlLoop::check_health(Snapshot& snap) {
  // The Go2's firmware cuts a motor that is too hot (2026-11-03: the robot stood still under the main policy,
  // went limp and fell on its chin). Lying down under control first is better than falling.
  int hot = 0;
  for (int j = 1; j < 12; ++j) {
    if (snap.motor_temp[j] > snap.motor_temp[hot]) hot = j;
  }
  const int t = snap.motor_temp[hot], soc = snap.battery_soc;
  const std::string& name = controller_->config().joint_names[hot];
  char buf[96];
  std::string fault;
  if (cfg_.motor_temp_lie_c > 0 && t >= cfg_.motor_temp_lie_c) {
    std::snprintf(buf, sizeof buf, "%s motor %d C >= %d C", name.c_str(), t, cfg_.motor_temp_lie_c);
    fault = buf;
  } else if (cfg_.battery_lie_pct > 0 && soc <= cfg_.battery_lie_pct) {
    std::snprintf(buf, sizeof buf, "battery %d%% <= %d%%", soc, cfg_.battery_lie_pct);
    fault = buf;
  }
  controller_->safety().set_external_fault(fault);

  // Warnings once per crossing, again after 3 C or 3% back.
  const bool hot_now = cfg_.motor_temp_warn_c > 0 && t >= cfg_.motor_temp_warn_c;
  const bool low_now = cfg_.battery_warn_pct > 0 && soc <= cfg_.battery_warn_pct;
  std::string warning;
  if (hot_now) {
    std::snprintf(buf, sizeof buf, "%s motor %d C (lies down at %d C)", name.c_str(), t, cfg_.motor_temp_lie_c);
    warning = buf;
    if (!warned_hot_) log(qc::LogLevel::Warn, "[Health] " + warning, snap);
  }
  if (low_now) {
    std::snprintf(buf, sizeof buf, "battery %d%%", soc);
    warning += (warning.empty() ? "" : ", ") + std::string(buf);
    if (!warned_battery_) log(qc::LogLevel::Warn, "[Health] " + std::string(buf), snap);
  }
  warned_hot_ = hot_now || (warned_hot_ && t > cfg_.motor_temp_warn_c - 3);
  warned_battery_ = low_now || (warned_battery_ && soc < cfg_.battery_warn_pct + 3);
  snap.health_warning = warning;
}

qc::RawState ControlLoop::raw_state(Snapshot& snap) {
  const LowState& s = low_state_;
  qc::RawState raw;
  raw.scan = std::move(scan_);
  scan_.reset();
  for (int j = 0; j < 12; ++j) {
    raw.q[j] = s.motor_state()[kSdkToRos[j]].q();
    raw.dq[j] = s.motor_state()[kSdkToRos[j]].dq();
  }
  const auto& imu = s.imu_state();
  raw.quat_wxyz = qc::Vec4(imu.quaternion()[0], imu.quaternion()[1], imu.quaternion()[2], imu.quaternion()[3]);
  // The gyro reads a few mrad/s while still (0.009 rad/s in pitch on this Go2); through the estimator's
  // omega x r_foot that was a steady 1.6 mm/s of false forward speed, and the odometry crept 10 cm a minute.
  const qc::Vec3 gyro(imu.gyroscope()[0], imu.gyroscope()[1], imu.gyroscope()[2]);
  if (!calib_done_) gyro_samples_.push_back(gyro);
  raw.gyro = gyro - gyro_offset_;
  raw.accel = qc::Vec3(imu.accelerometer()[0], imu.accelerometer()[1], imu.accelerometer()[2]);

  // Health, for /robot_state: the Go2's firmware cuts the motors when they are too hot or the battery is low.
  snap.battery_soc = s.bms_state().soc();
  snap.power_v = s.power_v();
  snap.power_a = s.power_a();
  for (int j = 0; j < 12; ++j) {
    const auto& m = s.motor_state()[kSdkToRos[j]];
    snap.motor_temp[j] = m.temperature();
    snap.motor_flags[j] = m.reserve()[0];
  }

  // foot_force is FR, FL, RR, RL; the controller's feet are FL, FR, RL, RR.
  const auto& ff = s.foot_force();
  const Eigen::Vector4d fsr(ff[1], ff[0], ff[3], ff[2]);
  if (!calib_done_) collect_fsr_sample(fsr, snap);
  snap.foot_force = fsr;
  qc::Vec4 contact;
  for (int i = 0; i < 4; ++i) contact[i] = (fsr[i] - fsr_offset_[i] > cfg_.contact_threshold) ? 1.0 : 0.0;
  raw.contact = contact;
  return raw;
}

void ControlLoop::collect_fsr_sample(const Eigen::Vector4d& fsr, Snapshot& snap) {
  calib_samples_.push_back(fsr);
  if (static_cast<int>(calib_samples_.size()) < calib_n_) return;

  Eigen::Vector4d mean = Eigen::Vector4d::Zero(), spread = Eigen::Vector4d::Zero();
  for (const auto& v : calib_samples_) mean += v;
  mean /= calib_samples_.size();
  for (const auto& v : calib_samples_) spread += (v - mean).cwiseAbs2();
  spread = (spread / calib_samples_.size()).cwiseSqrt();
  calib_samples_.clear();
  calib_done_ = true;
  set_gyro_zero(snap);

  // A foot carrying load reads high and steady, a moving one noisy: either way that zero is not a zero.
  static const char* names[4] = {"FL", "FR", "RL", "RR"};
  std::string line = "[RealDriver] FSR zero:";
  std::string rejected;
  char buf[96];
  for (int i = 0; i < 4; ++i) {
    std::snprintf(buf, sizeof(buf), "  %s=%.1f+-%.1f", names[i], mean[i], spread[i]);
    line += buf;
    const bool bad = mean[i] > cfg_.calib_offset_max || spread[i] > cfg_.calib_spread_max;
    if (bad) {
      std::snprintf(buf, sizeof(buf), "%s%s (%.1f+-%.1f)", rejected.empty() ? "" : ", ", names[i], mean[i], spread[i]);
      rejected += buf;
    } else {
      fsr_offset_[i] = mean[i];
    }
  }
  std::snprintf(buf, sizeof(buf), "  -> contact above %.0f counts over each", cfg_.contact_threshold);
  log(qc::LogLevel::Info, line + buf, snap);
  if (!rejected.empty()) {
    std::snprintf(buf, sizeof(buf), "(limits: offset<=%.0f, spread<=%.1f). Kept %.0f from config", cfg_.calib_offset_max,
                  cfg_.calib_spread_max, cfg_.fsr_bias);
    log(qc::LogLevel::Warn,
        "[RealDriver] Rejected FSR zero for " + rejected + " - loaded or moving during calibration " + buf +
            "; restart the driver with the feet unloaded to measure it.",
        snap);
  }
  snap.calibration_changed = true;
}

// The gyro's zero from the same still window as the FSR zero, kept only if the robot did not move in it.
void ControlLoop::set_gyro_zero(Snapshot& snap) {
  if (gyro_samples_.empty()) return;
  Eigen::Vector3d mean = Eigen::Vector3d::Zero(), spread = Eigen::Vector3d::Zero();
  for (const auto& v : gyro_samples_) mean += v;
  mean /= gyro_samples_.size();
  for (const auto& v : gyro_samples_) spread += (v - mean).cwiseAbs2();
  spread = (spread / gyro_samples_.size()).cwiseSqrt();
  gyro_samples_.clear();
  char buf[256];
  std::snprintf(buf, sizeof(buf), "[RealDriver] Gyro zero: %.4f %.4f %.4f rad/s (+-%.4f %.4f %.4f)", mean[0], mean[1],
                mean[2], spread[0], spread[1], spread[2]);
  if (mean.cwiseAbs().maxCoeff() > cfg_.gyro_offset_max || spread.maxCoeff() > cfg_.gyro_spread_max) {
    std::snprintf(buf + std::strlen(buf), sizeof(buf) - std::strlen(buf),
                  " rejected (limits %.2f, +-%.2f): the robot moved; restart the driver with it still.",
                  cfg_.gyro_offset_max, cfg_.gyro_spread_max);
    log(qc::LogLevel::Warn, buf, snap);
    return;
  }
  gyro_offset_ = mean;
  log(qc::LogLevel::Info, buf, snap);
}

void ControlLoop::send(const Eigen::Matrix<float, 12, 1>& targets) {
  qc::Controller& ctl = *controller_;
  const double max_torque = ctl.safety().active_max_torque();
  for (int i = 0; i < 12; ++i) {
    auto& m = low_cmd_.motor_cmd()[i];
    m.q() = targets[kRosToSdk[i]];
    m.dq() = 0.0f;
    m.tau() = 0.0f;
    if (max_torque <= 0.1) {
      // Damping only, no position hold, so the robot sinks under control. 6-11 are the rear legs.
      m.kp() = 0.0f;
      m.kd() = static_cast<float>(i >= 6 ? cfg_.emergency_kd_rear : cfg_.emergency_kd);
    } else {
      m.kp() = static_cast<float>(cfg_.kp);
      m.kd() = static_cast<float>(cfg_.kd);
    }
  }
  robot_.write(low_cmd_);
}

}  // namespace quadruped_real
