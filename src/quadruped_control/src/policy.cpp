#include "quadruped_control/policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace quadruped_control {

namespace {

constexpr float kUnitreeAngVelScale = 0.2f;
constexpr float kUnitreeJointVelScale = 0.05f;
constexpr double kTwoPi = 2.0 * M_PI;

// np.remainder / Python %: the result takes the divisor's sign.
double pymod(double a, double b) {
  double r = std::fmod(a, b);
  if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
  return r;
}

Vec4d pymod(const Vec4d& a, double b) { return a.unaryExpr([b](double v) { return pymod(v, b); }); }

Vec4d to_phase(const Vec4d& raw, const Vec4d& d) {
  Vec4d out;
  for (int i = 0; i < 4; ++i) {
    out[i] = raw[i] < d[i] ? raw[i] * (0.5 / d[i]) : 0.5 + (raw[i] - d[i]) * (0.5 / (1.0 - d[i]));
  }
  return out;
}

Vec4d to_raw(const Vec4d& phase, const Vec4d& d) {
  Vec4d out;
  for (int i = 0; i < 4; ++i) {
    out[i] = phase[i] < 0.5 ? phase[i] * (2.0 * d[i]) : d[i] + (phase[i] - 0.5) * (2.0 * (1.0 - d[i]));
  }
  return out;
}

double clock_ready(const Vec4d& offsets, const Vec4d& duty) {
  Eigen::Index lead;
  offsets.maxCoeff(&lead);
  return pymod(duty[lead] - offsets[lead], 1.0);
}

double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

void append(std::vector<float>& out, const Eigen::Ref<const VecXf>& v) {
  out.insert(out.end(), v.data(), v.data() + v.size());
}

template <typename Derived>
VecXf as_float(const Eigen::MatrixBase<Derived>& v) {
  return v.template cast<float>();
}

// ---- .qpol reading ----

template <typename T>
T read_pod(const std::string& bytes, size_t& pos) {
  if (pos + sizeof(T) > bytes.size()) throw std::runtime_error("truncated");
  T v;
  std::memcpy(&v, bytes.data() + pos, sizeof(T));
  pos += sizeof(T);
  return v;
}

Vec4d vec4(const YAML::Node& n) {
  if (n.IsScalar()) return Vec4d::Constant(n.as<double>());
  return Vec4d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>(), n[3].as<double>());
}

template <typename T>
std::optional<T> opt(const YAML::Node& n) {
  if (!n || n.IsNull()) return std::nullopt;
  return n.as<T>();
}

std::optional<Vec4d> opt_vec4(const YAML::Node& n) {
  if (!n || n.IsNull()) return std::nullopt;
  return vec4(n);
}

}  // namespace

ObsLayout layout_from_string(const std::string& name) {
  if (name == "isaac") return ObsLayout::Isaac;
  if (name == "unitree") return ObsLayout::Unitree;
  if (name == "unitree_vel") return ObsLayout::UnitreeVel;
  if (name == "wtw") return ObsLayout::Wtw;
  if (name == "unitree_clock") return ObsLayout::UnitreeClock;
  throw std::invalid_argument("[Policy] Unknown observation layout '" + name + "'.");
}

std::string to_string(ObsLayout layout) {
  switch (layout) {
    case ObsLayout::Isaac: return "isaac";
    case ObsLayout::Unitree: return "unitree";
    case ObsLayout::UnitreeVel: return "unitree_vel";
    case ObsLayout::Wtw: return "wtw";
    case ObsLayout::UnitreeClock: return "unitree_clock";
  }
  return "?";
}

PolicySpec PolicySpec::from_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("[Policy] Cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string bytes = ss.str();

  size_t pos = 0;
  if (bytes.compare(0, 4, "QPOL") != 0) throw std::runtime_error("[Policy] " + path + " is not a .qpol file.");
  pos = 4;
  const auto version = read_pod<uint32_t>(bytes, pos);
  if (version != 1) throw std::runtime_error("[Policy] " + path + ": unsupported .qpol version " + std::to_string(version));
  const auto meta_len = read_pod<uint32_t>(bytes, pos);
  if (pos + meta_len > bytes.size()) throw std::runtime_error("[Policy] " + path + ": truncated header.");
  const YAML::Node meta = YAML::Load(bytes.substr(pos, meta_len));
  pos += meta_len;

  std::map<std::string, std::pair<std::vector<int>, VecXf>> tensors;
  for (const auto& t : meta["tensors"]) {
    std::vector<int> shape = t["shape"].as<std::vector<int>>();
    size_t n = 1;
    for (int s : shape) n *= static_cast<size_t>(s);
    if (pos + n * sizeof(float) > bytes.size()) throw std::runtime_error("[Policy] " + path + ": truncated tensors.");
    VecXf data(static_cast<Eigen::Index>(n));
    std::memcpy(data.data(), bytes.data() + pos, n * sizeof(float));
    pos += n * sizeof(float);
    tensors[t["name"].as<std::string>()] = {shape, data};
  }
  auto tensor = [&](const std::string& name) -> const std::pair<std::vector<int>, VecXf>& {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("[Policy] " + path + ": missing tensor " + name);
    return it->second;
  };

  PolicySpec spec;
  spec.name = meta["name"].as<std::string>("");
  spec.source = meta["source"].as<std::string>("");
  spec.layout = layout_from_string(meta["layout"].as<std::string>());
  spec.obs_dim = meta["obs_dim"].as<int>();
  spec.obs_dim_single = meta["obs_dim_single"].as<int>();
  spec.history_len = meta["history_len"].as<int>();
  spec.action_scale = meta["action_scale"].as<float>(0.25f);
  const auto qpos = meta["desired_qpos"].as<std::vector<float>>();
  if (qpos.size() != 12) throw std::runtime_error("[Policy] " + path + ": desired_qpos must have 12 values.");
  for (int i = 0; i < 12; ++i) spec.desired_qpos[i] = qpos[i];

  std::vector<Mlp::Layer> layers;
  const int num_layers = meta["num_layers"].as<int>();
  for (int i = 0; i < num_layers; ++i) {
    const auto& w = tensor("layer" + std::to_string(i) + ".weight");
    const auto& b = tensor("layer" + std::to_string(i) + ".bias");
    Mlp::Layer l;
    l.weight = Eigen::Map<const MatXf>(w.second.data(), w.first.at(0), w.first.at(1));
    l.bias = b.second;
    layers.push_back(std::move(l));
  }
  spec.mlp = Mlp(tensor("obs_mean").second, tensor("obs_var").second, std::move(layers));

  if (const YAML::Node t = meta["trick"]) {
    spec.trick_clock_hz = t["clock_hz"].as<double>(0.0);
    spec.trick_switches = t["switches"].as<int>(1);
  }

  if (const YAML::Node w = meta["wtw"]) {
    WtwConfig c;
    c.blocks = w["blocks"].as<std::vector<int>>();
    const auto cmd = w["gait_command"].as<std::vector<float>>();
    c.gait_command = Eigen::Map<const VecXf>(cmd.data(), static_cast<Eigen::Index>(cmd.size()));
    for (const auto& g : w["gaits"]) {
      const auto v = g.second.as<std::vector<float>>();
      c.gaits[g.first.as<std::string>()] = Vec3f(v.at(0), v.at(1), v.at(2));
    }
    c.trained = w["trained"].as<std::vector<std::string>>();
    for (const auto& l : w["limits"]) {
      const auto v = l.second.as<std::vector<float>>();
      c.limits[l.first.as<int>()] = {v.at(0), v.at(1)};
    }
    c.stance = w["stance"].as<double>(0.5);
    c.max_stride = opt<double>(w["max_stride"]);
    c.yaw_radius = w["yaw_radius"].as<double>(0.3);
    spec.wtw = c;
  }

  if (const YAML::Node k = meta["clock"]) {
    SpeedClockConfig c;
    if (k["foot_offsets"] && !k["foot_offsets"].IsNull()) c.foot_offsets = vec4(k["foot_offsets"]);
    c.fast_offsets = opt_vec4(k["fast_offsets"]);
    c.fast_swing_times = opt_vec4(k["fast_swing_times"]);
    if (k["blend_speed"] && !k["blend_speed"].IsNull()) {
      c.blend_lo = k["blend_speed"][0].as<double>();
      c.blend_hi = k["blend_speed"][1].as<double>();
    }
    c.fixed_duty = opt_vec4(k["fixed_duty"]);
    c.reset_to_command = k["reset_to_command"].as<bool>(false);
    c.reset_speed = opt<double>(k["reset_speed"]);
    c.speed_from_measured = k["speed_source"].as<std::string>("command") == "measured";
    c.stride_min = k["stride_min"].as<double>();
    c.stride_gain = k["stride_gain"].as<double>();
    c.swing_time = k["swing_time"].as<double>();
    c.min_frequency = k["min_frequency"].as<double>();
    c.max_frequency = k["max_frequency"].as<double>();
    c.min_duty = k["min_duty"].as<double>(0.0);
    c.max_duty = k["max_duty"].as<double>(1.0);
    c.filter_time = k["filter_time"].as<double>();
    c.foot_x = opt<double>(k["foot_x"]);
    c.foot_y = opt<double>(k["foot_y"]);
    c.yaw_radius = k["yaw_radius"].as<double>(0.0);
    c.stand_threshold = k["stand_threshold"].as<double>();
    c.integrate_phase = k["integrate_phase"].as<bool>(false);
    c.sync_gain = k["sync_gain"].as<double>(4.0);
    c.max_foot_speed = k["max_foot_speed"].as<double>(0.0);
    spec.clock = c;
  }
  return spec;
}

PolicyRunner::PolicyRunner(PolicySpec spec) : spec_(std::move(spec)) {
  const std::string who = "[PolicyRunner] '" + spec_.name + "': ";
  if (spec_.mlp.input_dim() != spec_.obs_dim) throw std::invalid_argument(who + "network input does not match obs_dim.");
  if (spec_.mlp.output_dim() != 12) throw std::invalid_argument(who + "network must output 12 actions.");
  if (spec_.history_len * spec_.obs_dim_single != spec_.obs_dim) {
    throw std::invalid_argument(who + "obs_dim is not history_len x obs_dim_single.");
  }
  if (spec_.layout == ObsLayout::Wtw && !spec_.wtw) throw std::invalid_argument(who + "wtw layout without its gait settings.");
  if (spec_.layout == ObsLayout::UnitreeClock && !spec_.clock) throw std::invalid_argument(who + "clock layout without its clock settings.");
  if (spec_.wtw) gait_command_ = spec_.wtw->gait_command;
  history_ = MatXf::Zero(spec_.history_len, spec_.obs_dim_single);
  last_actions_ = VecXf::Zero(12);
  reset_history();
}

void PolicyRunner::reset_history() {
  history_.setZero();
  last_actions_.setZero();
  episode_start_ = true;
  gait_index_ = 0.0;
  reset_speed_clock();
}

VecXf PolicyRunner::infer(const ObsInput& in, const Eigen::VectorXd& commands, double dt) {
  policy_dt_ = dt;
  last_obs_ = build_obs(in, commands);
  last_actions_ = spec_.mlp.forward(last_obs_);
  return last_actions_;
}

VecXf PolicyRunner::build_obs(const ObsInput& in, const Eigen::VectorXd& commands) {
  const Vec3 proj_grav = rot_from_quat(in.quat_wxyz).transpose() * Vec3(0.0, 0.0, -1.0);
  const VecXf jpos_rel = as_float(in.q - spec_.desired_qpos.cast<double>());
  const VecXf jvel = as_float(in.dq);
  const VecXf gyro_scaled = VecXf(in.gyro.cast<float>()) * kUnitreeAngVelScale;
  const VecXf jvel_scaled = jvel * kUnitreeJointVelScale;

  auto cmd_head = [&](Eigen::Index n) {
    if (commands.size() < n) {
      throw std::invalid_argument("[PolicyRunner] '" + spec_.name + "': needs " + std::to_string(n) + " command values, got " +
                                  std::to_string(commands.size()) + ".");
    }
    return VecXf(commands.head(n).cast<float>());
  };

  std::vector<float> obs;
  obs.reserve(spec_.obs_dim_single);
  switch (spec_.layout) {
    case ObsLayout::Isaac: {
      append(obs, as_float(in.lin_vel));
      append(obs, as_float(in.gyro));
      append(obs, as_float(proj_grav));
      append(obs, cmd_head(std::max(0, spec_.obs_dim_single - 45)));
      append(obs, jpos_rel);
      append(obs, jvel);
      append(obs, last_actions_);
      break;
    }
    case ObsLayout::Unitree:
    case ObsLayout::UnitreeVel: {
      if (spec_.layout == ObsLayout::UnitreeVel) append(obs, as_float(in.lin_vel));
      append(obs, gyro_scaled);
      append(obs, as_float(proj_grav));
      append(obs, cmd_head(3));
      append(obs, jpos_rel);
      append(obs, jvel_scaled);
      append(obs, last_actions_);
      break;
    }
    case ObsLayout::Wtw: {
      const WtwConfig& w = *spec_.wtw;
      Vec3f velocity = cmd_head(3);
      if (w.max_stride && gait_command_.size() > 8) {
        const float sweep = std::hypot(velocity[0], velocity[1]) + std::abs(velocity[2]) * static_cast<float>(w.yaw_radius);
        const float limit = static_cast<float>(*w.max_stride) * gait_command_[1] / gait_command_[8];
        if (sweep > limit) velocity *= limit / sweep;
      }
      append(obs, as_float(in.lin_vel));
      append(obs, gyro_scaled);
      append(obs, as_float(proj_grav));
      append(obs, velocity);
      append(obs, jpos_rel);
      append(obs, jvel_scaled);
      append(obs, last_actions_);
      append(obs, gait_command_);
      append(obs, wtw_clock());
      break;
    }
    case ObsLayout::UnitreeClock: {
      const Vec3f velocity = limit_foot_speed(cmd_head(3));
      const Vec3f measured(static_cast<float>(in.lin_vel[0]), static_cast<float>(in.lin_vel[1]), static_cast<float>(in.gyro[2]));
      append(obs, as_float(in.lin_vel));
      append(obs, gyro_scaled);
      append(obs, as_float(proj_grav));
      append(obs, velocity);
      append(obs, jpos_rel);
      append(obs, jvel_scaled);
      append(obs, last_actions_);
      append(obs, speed_clock(velocity, measured));
      break;
    }
  }

  if (static_cast<int>(obs.size()) != spec_.obs_dim_single) {
    throw std::runtime_error("[PolicyRunner] '" + spec_.name + "': built a " + std::to_string(obs.size()) +
                             "-wide observation but the policy expects " + std::to_string(spec_.obs_dim_single) +
                             " (layout '" + to_string(spec_.layout) + "').");
  }
  const Eigen::Map<const Eigen::RowVectorXf> single(obs.data(), spec_.obs_dim_single);

  // Row 0 is the newest step; right after a reset every row is the current one.
  if (episode_start_) {
    history_.rowwise() = single;
    episode_start_ = false;
  } else {
    for (Eigen::Index r = history_.rows() - 1; r > 0; --r) history_.row(r) = history_.row(r - 1);
    history_.row(0) = single;
  }

  VecXf out(spec_.obs_dim);
  if (spec_.layout == ObsLayout::Wtw) {
    // Isaac Lab stacks history term by term, each term oldest to newest.
    Eigen::Index k = 0, col = 0;
    for (int width : spec_.wtw->blocks) {
      for (Eigen::Index r = history_.rows() - 1; r >= 0; --r) {
        out.segment(k, width) = history_.row(r).segment(col, width).transpose();
        k += width;
      }
      col += width;
    }
  } else {
    out = Eigen::Map<const VecXf>(history_.data(), spec_.obs_dim);
  }
  return out;
}

VecXf PolicyRunner::wtw_clock() {
  const VecXf& cmd = gait_command_;
  gait_index_ = pymod(gait_index_ + policy_dt_ * static_cast<double>(cmd[1]), 1.0);
  const double g = gait_index_;
  const double phase = cmd[2], offset = cmd[3], bound = cmd[4];
  const Vec4d raw = pymod(Vec4d(g + phase + offset + bound, g + offset, g + bound, g + phase), 1.0);
  const double d = cmd.size() > 8 ? static_cast<double>(cmd[8]) : spec_.wtw->stance;
  const Vec4d foot = to_phase(raw, Vec4d::Constant(d));
  return as_float(foot.unaryExpr([](double p) { return std::sin(kTwoPi * p); }));
}

double PolicyRunner::foot_sweep(const Vec3f& v) const {
  const SpeedClockConfig& c = *spec_.clock;
  if (!c.foot_x) {
    return std::hypot(v[0], v[1]) + std::abs(v[2]) * static_cast<float>(c.yaw_radius);
  }
  const double rx[4] = {*c.foot_x, *c.foot_x, -*c.foot_x, -*c.foot_x};
  const double fy = c.foot_y.value_or(0.0);
  const double ry[4] = {fy, -fy, fy, -fy};
  double best = -1.0;
  for (int i = 0; i < 4; ++i) {
    best = std::max(best, std::hypot(v[0] - v[2] * ry[i], v[1] + v[2] * rx[i]));
  }
  return best;
}

Vec3f PolicyRunner::limit_foot_speed(const Vec3f& velocity) const {
  const double max_speed = spec_.clock->max_foot_speed;
  if (max_speed <= 0.0) return velocity;
  const double sweep = foot_sweep(velocity);
  return velocity * static_cast<float>(std::min(1.0, max_speed / std::max(sweep, 1e-6)));
}

void PolicyRunner::clock_timing(Vec4d& offsets, Vec4d& duty, double& f) const {
  const SpeedClockConfig& c = *spec_.clock;
  const double s = clip((clock_speed_ - c.blend_lo) / std::max(c.blend_hi - c.blend_lo, 1e-6), 0.0, 1.0);
  offsets = c.foot_offsets + s * clock_offset_delta_;
  const Vec4d swing = clock_base_swing_ + s * clock_swing_delta_;
  const double stride = c.stride_min + c.stride_gain * clock_speed_;
  f = clip(clock_speed_ / stride, c.min_frequency, c.max_frequency);
  if (c.fixed_duty) {
    duty = *c.fixed_duty;
  } else {
    duty = (1.0 - swing.array() * f).max(c.min_duty).min(c.max_duty).matrix();
  }
}

void PolicyRunner::reset_speed_clock() {
  if (!spec_.clock) return;
  const SpeedClockConfig& c = *spec_.clock;
  const Vec4d fast = c.fast_offsets.value_or(c.foot_offsets);
  clock_offset_delta_ = pymod((fast - c.foot_offsets).array() + 0.5, 1.0).array() - 0.5;
  clock_base_swing_ = Vec4d::Constant(c.swing_time);
  clock_swing_delta_ = c.fast_swing_times.value_or(clock_base_swing_) - clock_base_swing_;
  clock_speed_ = 0.0;
  Vec4d offsets, duty;
  double f;
  clock_timing(offsets, duty, f);
  clock_index_ = clock_ready(offsets, duty);
  clock_phase_ = to_phase(pymod((offsets.array() + clock_index_).matrix(), 1.0), duty);
  clock_fresh_ = true;
}

Vec4d PolicyRunner::clock_integrate(const Vec4d& offsets, const Vec4d& d, double f, bool standing, double dt) {
  const SpeedClockConfig& c = *spec_.clock;
  Vec4d phase = clock_phase_;
  const Vec4d raw = to_raw(phase, d);
  double sin_sum = 0.0, cos_sum = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double angle = kTwoPi * (raw[i] - offsets[i]);
    sin_sum += std::sin(angle);
    cos_sum += std::cos(angle);
  }
  const double index = std::atan2(sin_sum / 4.0, cos_sum / 4.0) / kTwoPi;
  bool any_swing = false;
  for (int i = 0; i < 4; ++i) {
    const double error = pymod(index + offsets[i] - raw[i] + 0.5, 1.0) - 0.5;
    const double sync = clip(1.0 + c.sync_gain * error, 0.5, 1.5);
    const bool swinging = phase[i] > 0.5 + 1e-4;
    double rate = swinging ? 0.5 * f / (1.0 - d[i]) : 0.5 * f / d[i] * sync;
    if (standing && !swinging) rate = 0.0;
    phase[i] = pymod(phase[i] + rate * dt, 1.0);
    any_swing = any_swing || phase[i] > 0.5 + 1e-4;
  }
  if (standing && !any_swing) {
    clock_index_ = clock_ready(offsets, d);
    phase = to_phase(pymod((offsets.array() + clock_index_).matrix(), 1.0), d);
  } else {
    clock_index_ = pymod(index, 1.0);
  }
  clock_phase_ = phase;
  return phase;
}

VecXf PolicyRunner::speed_clock(const Vec3f& velocity, const Vec3f& measured) {
  const SpeedClockConfig& c = *spec_.clock;
  const double dt = policy_dt_;
  const double sweep = foot_sweep(c.speed_from_measured ? measured : velocity);
  if (clock_fresh_ && c.reset_speed) {
    clock_speed_ = *c.reset_speed;
  } else if (clock_fresh_ && c.reset_to_command) {
    clock_speed_ = sweep;
  }
  clock_fresh_ = false;
  clock_speed_ += std::min(dt / c.filter_time, 1.0) * (sweep - clock_speed_);
  Vec4d offsets, d;
  double f;
  clock_timing(offsets, d, f);
  const bool standing = velocity.norm() < c.stand_threshold;
  Vec4d foot;
  if (c.integrate_phase) {
    foot = clock_integrate(offsets, d, f, standing, dt);
  } else {
    clock_index_ = standing ? clock_ready(offsets, d) : pymod(clock_index_ + dt * f, 1.0);
    foot = to_phase(pymod((offsets.array() + clock_index_).matrix(), 1.0), d);
  }
  VecXf out(8);
  for (int i = 0; i < 4; ++i) {
    out[i] = static_cast<float>(std::sin(kTwoPi * foot[i]));
    out[4 + i] = static_cast<float>(std::cos(kTwoPi * foot[i]));
  }
  return out;
}

void PolicyRunner::set_gait_vector(const VecXf& values) {
  if (!spec_.wtw) return;
  const WtwConfig& w = *spec_.wtw;
  VecXf command = gait_command_;
  const Eigen::Index n = std::min(values.size(), command.size());
  command.head(n) = values.head(n);
  for (const auto& [column, range] : w.limits) {
    if (column < command.size()) command[column] = std::min(std::max(command[column], range.first), range.second);
  }
  const Vec3f phases = command.segment<3>(2);
  for (const auto& [name, preset] : w.gaits) {
    if (std::find(w.trained.begin(), w.trained.end(), name) != w.trained.end()) continue;
    const bool close = ((phases - preset).cwiseAbs().array() <= 1e-3f + 1e-5f * preset.cwiseAbs().array()).all();
    if (close) command.segment<3>(2) = w.gaits.at("trot");
  }
  float best = INFINITY;
  Vec3f snap = command.segment<3>(2);
  for (const auto& name : w.trained) {
    const Vec3f& preset = w.gaits.at(name);
    // Summed left to right as numpy does: equidistant gaits (trot and bound are, for some inputs)
    // must break the tie the same way.
    float dist = 0.0f;
    for (int i = 0; i < 3; ++i) dist += std::abs(preset[i] - command[2 + i]);
    if (dist < best) {
      best = dist;
      snap = preset;
    }
  }
  command.segment<3>(2) = snap;
  gait_command_ = command;
}

void PolicyRunner::set_gait(const std::optional<std::string>& gait, const std::map<std::string, float>& fields) {
  if (!spec_.wtw) return;
  static const std::map<std::string, int> kColumns = {{"height", 0}, {"frequency", 1}, {"swing", 5},
                                                      {"pitch", 6},  {"width", 7},     {"duty", 8}};
  VecXf command = gait_command_;
  if (gait) {
    auto it = spec_.wtw->gaits.find(*gait);
    if (it == spec_.wtw->gaits.end()) throw std::invalid_argument("[PolicyRunner] Unknown gait '" + *gait + "'.");
    command.segment<3>(2) = it->second;
  }
  for (const auto& [name, value] : fields) {
    const int column = kColumns.at(name);
    if (column >= command.size()) throw std::out_of_range("[PolicyRunner] This policy has no '" + name + "' gait column.");
    command[column] = value;
  }
  set_gait_vector(command);
}

}  // namespace quadruped_control
