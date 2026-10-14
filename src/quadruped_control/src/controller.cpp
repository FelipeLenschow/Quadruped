#include "quadruped_control/controller.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <set>

namespace quadruped_control {

namespace {

std::string quoted_or_none(const std::optional<std::string>& s) { return s ? *s : "None"; }

std::string list(const std::vector<std::string>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) out += (i ? ", '" : "'") + items[i] + "'";
  return out + "]";
}

}  // namespace

// ---------------------------------------------------------------------------
// Command slew
// ---------------------------------------------------------------------------

CommandSlewLimiter::CommandSlewLimiter(double lin_accel, double lin_decel, double yaw_accel, double yaw_decel)
    : accel_(lin_accel, lin_accel, yaw_accel), decel_(lin_decel, lin_decel, yaw_decel) {
  enabled_ = (accel_.array() > 0).any() || (decel_.array() > 0).any();
}

Eigen::VectorXd CommandSlewLimiter::operator()(const Eigen::VectorXd& cmd, double dt) {
  if (!enabled_ || cmd.size() < 3) return cmd;
  Vec3 next;
  for (int i = 0; i < 3; ++i) {
    const double target = cmd[i], cur = current_[i];
    const bool reversing = cur * target < 0.0;
    const bool slowing = reversing || std::abs(target) < std::abs(cur);
    const double goal = reversing ? 0.0 : target;
    const double limit = slowing ? decel_[i] : accel_[i];
    const double step = std::clamp(goal - cur, -limit * dt, limit * dt);
    next[i] = limit > 0.0 ? cur + step : target;
  }
  current_ = next;
  Eigen::VectorXd out = cmd;
  out.head<3>() = current_;
  return out;
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

Controller::Controller(ControllerConfig cfg)
    : cfg_(std::move(cfg)),
      estimator_(Kinematics(cfg_.geometry), cfg_.estimator),
      safety_(cfg_, events_),
      pose_(cfg_, events_),
      slew_(cfg_.lin_accel, cfg_.lin_decel, cfg_.yaw_accel, cfg_.yaw_decel),
      desired_qpos_(default_stance()),
      lying_cos_(std::cos(cfg_.lying_tilt_deg * M_PI / 180.0)),
      latest_targets_(default_stance()),
      transition_duration_(cfg_.mode_transition_duration),
      transition_start_targets_(default_stance()) {}

Controller::Loaded* Controller::find(const std::string& name) {
  for (auto& p : policies_) {
    if (p.name == name) return &p;
  }
  return nullptr;
}

const Controller::Loaded* Controller::find(const std::string& name) const {
  for (const auto& p : policies_) {
    if (p.name == name) return &p;
  }
  return nullptr;
}

PolicyRunner* Controller::runner(const std::string& name) {
  Loaded* p = find(name);
  return p ? &p->runner : nullptr;
}

void Controller::set_catalog(std::vector<std::pair<std::string, PolicyEntry>> catalog) { catalog_ = std::move(catalog); }

void Controller::add_policy(const std::string& name, PolicySpec spec, const PolicyEntry& entry) {
  add_policy(name, PolicyRunner(std::move(spec)), entry);
}

void Controller::add_policy(const std::string& name, PolicyRunner runner, const PolicyEntry& entry) {
  if (Loaded* p = find(name)) {
    p->runner = std::move(runner);
    p->entry = entry;
  } else {
    policies_.push_back({name, std::move(runner), entry});
  }
  load_errors_.erase(name);
}

void Controller::unload_policy(const std::string& name) {
  policies_.erase(std::remove_if(policies_.begin(), policies_.end(), [&](const Loaded& p) { return p.name == name; }),
                  policies_.end());
}

void Controller::set_load_error(const std::string& name, const std::string& error) { load_errors_[name] = error; }

PolicyEntry Controller::entry(const std::string& name) const {
  if (const Loaded* p = find(name)) return p->entry;
  for (const auto& [n, e] : catalog_) {
    if (n == name) return e;
  }
  return PolicyEntry{};
}

std::vector<std::string> Controller::available() const {
  std::vector<std::string> out;
  if (has_policy("main")) out.push_back("main");
  for (const auto& [n, e] : catalog_) {
    if (n == "main" || (main_alias_ && has_policy("main") && n == *main_alias_)) continue;
    out.push_back(n);
  }
  return out;
}

std::vector<std::string> Controller::loaded() const {
  std::vector<std::string> out;
  for (const auto& p : policies_) out.push_back(p.name);
  return out;
}

PolicyRequest Controller::request_policy(const std::string& requested) {
  const std::string name = (main_alias_ && has_policy("main") && requested == *main_alias_) ? "main" : requested;
  if (safety_.is_estopped() || safety_.is_numb() || safety_.is_policy_blocked()) {
    events_.log(LogLevel::Warn, "[Pipeline] Refusing policy '" + name + "': e-stop, numb or the safety switch is on.");
    return PolicyRequest::Refused;
  }
  if (active_ && name == *active_) {
    pending_.reset();
    return PolicyRequest::AlreadyActive;
  }
  const bool loaded_now = has_policy(name);
  const bool in_catalog = std::any_of(catalog_.begin(), catalog_.end(), [&](const auto& c) { return c.first == name; });
  if (!loaded_now && !in_catalog) {
    events_.log(LogLevel::Warn, "[Pipeline] Unknown policy '" + name + "'. Available: " + list(available()));
    return PolicyRequest::Unknown;
  }
  if (!loaded_now) load_errors_.erase(name);
  pending_ = name;
  policy_error_.reset();
  events_.log(LogLevel::Info, "[Pipeline] Switching to policy '" + name + "'...");
  return loaded_now ? PolicyRequest::Ready : PolicyRequest::NeedsLoad;
}

void Controller::advance_policy_switch() {
  if (!pending_) return;
  const std::string name = *pending_;
  auto err = load_errors_.find(name);
  if (err != load_errors_.end()) {
    policy_error_ = name + ": " + err->second;
    events_.log(LogLevel::Error, "[Pipeline] Policy switch failed: " + *policy_error_);
    pending_.reset();
    return;
  }
  Loaded* p = find(name);
  if (!p) return;
  const std::optional<std::string> old = active_;
  p->runner.reset_history();
  active_ = name;
  pending_.reset();
  if (mode_ == "policy") {
    transition_active_ = true;
    transition_start_time_.reset();
    transition_start_targets_ = latest_targets_;
    transition_duration_ = cfg_.policy_switch_blend;
  }
  const std::set<std::string> keep = {name, "main", "pose", safety_name_};
  for (const auto& stale : loaded()) {
    if (!keep.count(stale)) unload_policy(stale);
  }
  events_.log(LogLevel::Info, "[Pipeline] Policy '" + quoted_or_none(old) + "' -> '" + name + "', protections " +
                                  (entry(name).protections ? list(protection_names(entry(name).protections)) : "none") + ".");
}

// ---- inputs ----

void Controller::set_mode(const std::string& raw_mode) {
  std::string m;
  for (char c : raw_mode) {
    if (!std::isspace(static_cast<unsigned char>(c))) m += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (m == "policy" && safety_.is_estopped()) {
    events_.log(LogLevel::Warn, "[Pipeline] Refusing policy mode: E-STOP is latched. Release it at the robot first.");
    return;
  }
  if (m == "policy" && safety_.is_numb()) {
    events_.log(LogLevel::Warn, "[Pipeline] Refusing policy mode: the robot is numb. Release it first.");
    return;
  }
  if (m != "pose" && m != "policy") {
    events_.log(LogLevel::Warn, "[Pipeline] Unknown mode '" + m + "'. Use 'pose' or 'policy'.");
    return;
  }
  if (m == mode_) return;
  events_.log(LogLevel::Info, "[Pipeline] Mode switched: " + mode_ + " → " + m);
  mode_ = m;
  transition_active_ = true;
  transition_start_time_.reset();
  transition_duration_ = cfg_.mode_transition_duration;
  const bool was_limp = safety_.active_max_torque() <= 0.1;
  if (was_limp && last_measured_q_) {
    transition_start_targets_ = *last_measured_q_;
    events_.log(LogLevel::Info, "[Pipeline] Torque was off - blending from measured joints.");
  } else {
    transition_start_targets_ = latest_targets_;
  }
  // The pose generator is the way out of an unsafe state, so entering pose mode clears the trip.
  if (m == "pose") {
    safety_.clear_block(true);
    safety_elapsed_.reset();
    pose_.sync_to_current(transition_start_targets_);
  }
}

void Controller::cmd_vel(double vx, double vy, double wz, double now) {
  cmd_ = Eigen::Vector4d(vx, vy, wz, 0.0);
  cmd_time_ = now;
}

std::optional<double> Controller::cmd_age(double now) const {
  if (!cmd_time_) return std::nullopt;
  return now - *cmd_time_;
}

Eigen::Vector4d Controller::current_cmd_vel(double now) const {
  const auto age = cmd_age(now);
  if (!age || *age > cfg_.cmd_timeout) return Eigen::Vector4d::Zero();
  return cmd_;
}

void Controller::trick(const std::vector<double>& values, double now) {
  for (int i = 0; i < 3; ++i) trick_[i] = i < static_cast<int>(values.size()) ? values[i] : 0.0;
  trick_time_ = now;
}

std::array<double, 3> Controller::trick_command(double now) const {
  if (!trick_time_ || now - *trick_time_ > cfg_.cmd_timeout) return {0.0, 0.0, 0.0};
  return trick_;
}

std::vector<double> Controller::trick_input(const PolicyEntry& e, double now) {
  const auto command = trick_command(now);
  if (!e.clock_hz) return {command[0], command[1], command[2]};
  if (command[0] < 0.5) {
    trick_phase_ = 0.0;
    return {0.0, 0.0, 0.0};
  }
  trick_phase_ += 2.0 * M_PI * e.clock_hz * cfg_.policy_dt;
  return {1.0, std::sin(trick_phase_), std::cos(trick_phase_)};
}

const VecXf* Controller::gait_command() const {
  for (const auto& p : policies_) {
    if (p.runner.spec().layout == ObsLayout::Wtw) return &p.runner.gait_command();
  }
  return nullptr;
}

void Controller::gait(const VecXf& values) {
  if (values.size() != 8 && values.size() != 9) return;
  gait_base_ = values;
  if (gait_override_time_) return;
  for (auto& p : policies_) {
    if (p.runner.spec().layout == ObsLayout::Wtw) p.runner.set_gait_vector(values);
  }
}

void Controller::gait_override(const VecXf& values, double now) {
  if (values.size() != 8 && values.size() != 9) return;
  const VecXf* current = gait_command();
  if (!gait_override_time_ && !gait_base_ && current) gait_base_ = *current;
  gait_override_time_ = now;
  for (auto& p : policies_) {
    if (p.runner.spec().layout == ObsLayout::Wtw) p.runner.set_gait_vector(values);
  }
}

void Controller::release_gait_override(double now) {
  if (!gait_override_time_ || now - *gait_override_time_ <= cfg_.cmd_timeout) return;
  gait_override_time_.reset();
  if (!gait_base_) return;
  for (auto& p : policies_) {
    if (p.runner.spec().layout == ObsLayout::Wtw) p.runner.set_gait_vector(*gait_base_);
  }
}

// ---- control ----

void Controller::process_state(const RawState& raw, bool update_estimator) {
  State s;
  s.quat_wxyz = raw.quat_wxyz;
  s.gyro = raw.gyro;
  if (raw.accel) s.accel = *raw.accel;
  s.q = raw.q;
  s.dq = raw.dq;
  s.has_pos = raw.pos.has_value();
  if (raw.pos) s.base_pos = *raw.pos;
  if (raw.contact) s.feet_contact = *raw.contact;
  s.base_lin_vel_sim = raw.vel.value_or(Vec3::Zero());
  s.base_lin_vel_est = update_estimator ? estimator_.update(s.quat_wxyz, s.accel, s.feet_contact, s.q, s.dq, s.gyro)
                                        : estimator_.velocity();
  s.base_lin_vel = cfg_.use_estimator ? s.base_lin_vel_est : s.base_lin_vel_sim;
  state_ = s;
}

Vec12 Controller::step_policy(const std::string& name, const Eigen::VectorXd& commands) {
  Loaded* p = find(name);
  const Eigen::VectorXd cmd = p->entry.commands == CommandMode::Zero ? Eigen::VectorXd::Zero(commands.size()) : commands;
  const VecXf raw = p->runner.infer(ObsInput{state_.quat_wxyz, state_.base_lin_vel, state_.gyro, state_.q, state_.dq},
                                    cmd, cfg_.policy_dt);
  const PolicySpec& spec = p->runner.spec();
  Vec12 targets;
  for (int i = 0; i < 12; ++i) targets[i] = raw[i] * spec.action_scale + spec.desired_qpos[i];
  return targets;
}

bool Controller::is_lying() const {
  const Vec3 g = projected_gravity_b(state_.quat_wxyz);
  const bool level = -g[2] / std::max(g.norm(), 1e-6) >= lying_cos_;
  float err = 0.0f, max_dq = 0.0f;
  for (int i = 0; i < 12; ++i) {
    err += std::abs(static_cast<float>(state_.q[i]) - static_cast<float>(cfg_.lie_pose[i]));
    max_dq = std::max(max_dq, std::abs(static_cast<float>(state_.dq[i])));
  }
  return level && err / 12.0f < cfg_.lying_pose_err && max_dq < cfg_.lying_joint_vel;
}

void Controller::watch_safety_policy() {
  if (safety_.is_numb() || !cfg_.safety_switch) return;
  if (!has_policy(safety_name_)) {
    safety_.set_numb("safety switch (" + safety_.trip_reason() + "), no safety policy");
    return;
  }
  *safety_elapsed_ += cfg_.policy_dt;
  lying_for_ = is_lying() ? lying_for_ + cfg_.policy_dt : 0.0;
  if (lying_for_ >= cfg_.lying_hold) {
    safety_.set_numb("lying after the safety switch");
  } else if (*safety_elapsed_ >= cfg_.safety_policy_timeout) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "safety policy ran %.0f s without lying still", cfg_.safety_policy_timeout);
    safety_.set_numb(buf);
  }
}

StepResult Controller::step(const RawState& raw, const std::optional<Eigen::VectorXd>& cmd_vel_in, double sim_time,
                            double now) {
  const bool policy_step = step_counter_ % cfg_.decimation == 0;
  ++step_counter_;
  Eigen::VectorXd cmd_vel = cmd_vel_in ? *cmd_vel_in : Eigen::VectorXd(current_cmd_vel(now));

  process_state(raw, policy_step);
  last_measured_q_ = state_.q.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });

  StepResult result;
  if (policy_step) {
    // Torque comes back on a robot that lay limp: hold where it is instead of snapping to the old pose.
    if (was_estopped_ && !safety_.is_estopped()) {
      was_estopped_ = false;
      mode_ = "pose";
      transition_active_ = false;
      pose_.sync_to_current();
      events_.log(LogLevel::Warn, "[Pipeline] E-STOP released — holding at measured joint positions in POSE mode.");
    }
    if (was_numb_ && !safety_.is_numb()) {
      was_numb_ = false;
      mode_ = "pose";
      transition_active_ = false;
      safety_elapsed_.reset();
      pose_.sync_to_current();
      events_.log(LogLevel::Warn, "[Pipeline] Numb released: holding at measured joint positions in POSE mode.");
    }

    advance_policy_switch();
    release_gait_override(now);

    if (mode_ != "policy" || safety_.is_estopped()) {
      slew_.reset();
      slew_time_.reset();
    }

    Vec12 final_targets;
    double max_torque;
    if (mode_ == "pose") {
      // No ROM or tilt checks here: the pose generator is how the robot gets out of an unsafe state.
      const bool heartbeat_ok = safety_.heartbeat_ok(now);
      if (heartbeat_ok && !pose_heartbeat_was_ok_) {
        pose_.sync_to_current();
        events_.log(LogLevel::Info, "[Pipeline] Heartbeat restored in pose mode — synced to current joint positions.");
      }
      pose_heartbeat_was_ok_ = heartbeat_ok;
      if (heartbeat_ok) {
        final_targets = safety_.clip_soft(pose_.step(state_.q, sim_time));
        result.pose_status = pose_.status();
        max_torque = safety_.global_max_torque();
      } else {
        final_targets = desired_qpos_;
        max_torque = 0.0;
      }
      safety_.set_active_max_torque(max_torque);
    } else {
      const uint8_t protections = active_ ? entry(*active_).protections : static_cast<uint8_t>(kAllProtections);
      const bool is_safe = safety_.evaluate(state_, protections, now).first;

      double dt = cfg_.policy_dt;
      if (slew_time_ && sim_time - *slew_time_ > 0.0 && sim_time - *slew_time_ <= 0.1) dt = sim_time - *slew_time_;
      slew_time_ = sim_time;
      cmd_vel = slew_(cmd_vel, dt);

      if (!safety_.is_policy_blocked()) safety_elapsed_.reset();
      const bool takeover = !is_safe || safety_.is_policy_blocked();
      std::optional<Vec12> main, safety;
      // Under an e-stop nothing runs, the safety policy included.
      if (takeover && has_policy(safety_name_) && !safety_.is_numb() && !safety_.is_estopped()) {
        if (!safety_elapsed_) {
          safety_elapsed_ = 0.0;
          lying_for_ = 0.0;
          transition_active_ = false;
          find(safety_name_)->runner.reset_history();
          events_.log(LogLevel::Error,
                      "[Pipeline] Safety policy '" + safety_name_ + "' takes over from '" + quoted_or_none(active_) + "'.");
        }
        safety = step_policy(safety_name_, cmd_vel);
      } else if (!takeover && active_ && has_policy(*active_)) {
        Eigen::VectorXd command = cmd_vel;
        const PolicyEntry e = entry(*active_);
        if (e.commands == CommandMode::Trick) {
          const auto t = trick_input(e, now);
          command = Eigen::Vector4d(t[0], t[1], t[2], 0.0);
        }
        main = step_policy(*active_, command);
      }

      std::tie(final_targets, max_torque) = safety_.process(main, safety, state_, protections, now);
      if (safety_.is_policy_blocked()) {
        if (!safety_elapsed_) {
          safety_elapsed_ = 0.0;
          lying_for_ = 0.0;
        }
        watch_safety_policy();
      } else {
        safety_elapsed_.reset();
      }
    }

    // A stopped robot holds the last command, at zero torque: no jump if torque comes back.
    if (safety_.is_estopped()) {
      was_estopped_ = true;
      final_targets = latest_targets_;
      max_torque = 0.0;
      transition_active_ = false;
    } else if (safety_.is_numb()) {
      was_numb_ = true;
      final_targets = latest_targets_;
      max_torque = 0.0;
      transition_active_ = false;
    }

    if (transition_active_) {
      if (!transition_start_time_) transition_start_time_ = sim_time;
      const double alpha = std::clamp((sim_time - *transition_start_time_) / transition_duration_, 0.0, 1.0);
      final_targets = (1.0 - alpha) * transition_start_targets_ + alpha * final_targets;
      if (alpha >= 1.0) {
        transition_active_ = false;
        transition_start_time_.reset();
      }
    }

    latest_targets_ = final_targets;
    last_command_ = cmd_vel;
    result.max_torque = max_torque;
  } else {
    result.max_torque = safety_.active_max_torque();
  }
  result.targets = latest_targets_;
  result.policy_step = policy_step;
  return result;
}

void Controller::reset() {
  for (auto& p : policies_) p.runner.reset_history();
  slew_.reset();
  slew_time_.reset();
  latest_targets_ = desired_qpos_;
  step_counter_ = 0;
  transition_active_ = false;
  transition_start_time_.reset();
  transition_start_targets_ = desired_qpos_;
  safety_elapsed_.reset();
  lying_for_ = 0.0;
  pending_.reset();
  trick_phase_ = 0.0;
}

}  // namespace quadruped_control
