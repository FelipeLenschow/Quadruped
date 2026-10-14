#include "driver_node.hpp"

#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <ament_index_cpp/get_package_prefix.hpp>

#include "config_loader.hpp"
#include "json.hpp"

extern char** environ;

namespace quadruped_real {

namespace {

const char* kRed = "\033[91m";
const char* kYellow = "\033[93m";
const char* kBold = "\033[1m";
const char* kReset = "\033[0m";
const char* kGaitNames[9] = {"height", "frequency", "phase", "offset", "bound", "swing", "pitch", "width", "duty"};
const std::vector<std::pair<std::string, Eigen::Vector3f>> kWtwGaits = {
    {"trot", {0.5f, 0.0f, 0.0f}}, {"pace", {0.0f, 0.0f, 0.5f}}, {"bound", {0.0f, 0.5f, 0.0f}},
    {"pronk", {0.0f, 0.0f, 0.0f}}, {"walk", {0.0f, 0.75f, 0.5f}}};

bool exists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}

double mtime(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 ? st.st_mtim.tv_sec + st.st_mtim.tv_nsec * 1e-9 : 0.0;
}

std::string trim(const std::string& s) {
  const auto a = s.find_first_not_of(" \t\r\n");
  const auto b = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

double r3(double v) { return std::round(v * 1000.0) / 1000.0; }

void print_shutdown_banner(const std::string& reason) {
  std::cout << "\r" << std::string(120, ' ') << "\r";
  std::cout << "\n" << kBold << kRed << "╔══════════════════════════════════════════════════════╗\n"
            << "║  ⚠  COMMAND SAFETY PROCESSOR — EMERGENCY SHUTDOWN ⚠  ║\n"
            << "╠══════════════════════════════════════════════════════╣\n";
  char buf[160];
  std::snprintf(buf, sizeof(buf), "║  Reason: %-43s ║\n", reason.substr(0, 43).c_str());
  std::cout << buf;
  for (size_t i = 43; i < reason.size(); i += 49) {
    std::snprintf(buf, sizeof(buf), "║    %-49s ║\n", reason.substr(i, 49).c_str());
    std::cout << buf;
  }
  std::cout << "║                                                      ║\n"
            << "║  ➜  Main policy BLOCKED.                             ║\n"
            << "║  ➜  Press [ENTER] here to RESET and re-enable.       ║\n"
            << "╚══════════════════════════════════════════════════════╝" << kReset << "\n"
            << std::endl;
}

void print_estop_banner() {
  std::cout << "\r" << std::string(120, ' ') << "\r";
  std::cout << "\n" << kBold << kRed << "╔══════════════════════════════════════════════════════╗\n"
            << "║  ⛔  E-STOP  —  torque cut to 0 Nm                   ║\n"
            << "╠══════════════════════════════════════════════════════╣\n"
            << "║  ➜  Latched. No policy, no pose, no recovery run.    ║\n"
            << "║  ➜  Press [ENTER] HERE, on the robot, to release.    ║\n"
            << "║     The console cannot clear this.                   ║\n"
            << "╚══════════════════════════════════════════════════════╝" << kReset << "\n"
            << std::endl;
}

geometry_msgs::msg::Vector3 vec3(double x, double y, double z) {
  geometry_msgs::msg::Vector3 v;
  v.x = x, v.y = y, v.z = z;
  return v;
}

geometry_msgs::msg::Quaternion quat(double w, double x, double y, double z) {
  geometry_msgs::msg::Quaternion q;
  q.w = w, q.x = x, q.y = y, q.z = z;
  return q;
}

builtin_interfaces::msg::Time stamp(double seconds) {
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(std::floor(seconds));
  t.nanosec = static_cast<uint32_t>((seconds - std::floor(seconds)) * 1e9);
  return t;
}

}  // namespace

// ---------------------------------------------------------------------------
// Heartbeat statistics
// ---------------------------------------------------------------------------

void HeartbeatStats::trim(double now) {
  while (!gaps_.empty() && std::get<0>(gaps_.front()) < now - 10.0) gaps_.pop_front();
}

void HeartbeatStats::add(double now) {
  ++count_;
  if (last_) {
    const double gap = now - *last_;
    double period = 0.0;
    if (gaps_.size() >= 5) {
      std::vector<double> g;
      for (const auto& e : gaps_) g.push_back(std::get<1>(e));
      std::sort(g.begin(), g.end());
      const size_t n = g.size();
      period = n % 2 ? g[n / 2] : 0.5 * (g[n / 2 - 1] + g[n / 2]);
    }
    const long missed = period > 0.0 ? std::max(0L, static_cast<long>(std::nearbyint(gap / period)) - 1) : 0L;
    gaps_.emplace_back(now, gap, missed);
    missed_total_ += missed;
  } else {
    first_ = now;
  }
  last_ = now;
  trim(now);
}

std::string HeartbeatStats::summary(double now) {
  if (!last_) return "null";
  trim(now);
  double max_gap = now - *last_;
  long missed_10s = 0;
  for (const auto& e : gaps_) {
    max_gap = std::max(max_gap, std::get<1>(e));
    missed_10s += std::get<2>(e);
  }
  const double span = std::min(10.0, now - *first_);
  return json::Object()
      .add("hz", json::num(span > 0 ? std::round(gaps_.size() / span * 10.0) / 10.0 : 0.0))
      .add("missed_10s", json::integer(missed_10s))
      .add("missed", json::integer(missed_total_))
      .add("count", json::integer(count_))
      .add("max_gap_ms", json::integer(std::lround(max_gap * 1000.0)))
      .dump();
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

DriverNode::DriverNode(Paths paths, YAML::Node config, ControlLoop& loop, CommandBox& commands, SnapshotBox& snapshots)
    : rclcpp::Node("real_driver"),
      paths_(std::move(paths)),
      config_(std::move(config)),
      loop_(loop),
      commands_(commands),
      snapshots_(snapshots),
      kinematics_(qc::LegGeometry::from_yaml(paths_.geometry)) {
  joint_names_ = qc::ControllerConfig().joint_names;
  state_period_ = 1.0 / get<double>(config_["cmd_vel"], "robot_state_rate", 10.0);
  odom_height_ = get<double>(config_["control"], "default_height", 0.33);
  have_calibration_ = true;

  // policies.json, as PolicyManager reads it.
  YAML::Node policies;
  try {
    policies = YAML::LoadFile(paths_.policies);
  } catch (const std::exception& e) {
    RCLCPP_WARN(get_logger(), "[ConfigLoader] %s not readable (%s). No policies to switch to, no safety policy.",
                paths_.policies.c_str(), e.what());
  }
  safety_policy_ = get<std::string>(policies, "safety_policy", "");
  for (const auto& item : policies["policies"]) {
    const std::string name = item.first.as<std::string>();
    const YAML::Node spec = item.second;
    CatalogEntry c;
    c.path = get<std::string>(spec, "path", "");
    try {
      c.entry.protections = spec["protections"] ? qc::protections_from_names(spec["protections"].as<std::vector<std::string>>())
                                                : static_cast<uint8_t>(qc::kAllProtections);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "[Policies] Policy '%s': %s. Using all of them.", name.c_str(), e.what());
      c.entry.protections = qc::kAllProtections;
    }
    const YAML::Node cmd = spec["commands"];
    if (!cmd) {
      c.entry.commands = qc::CommandMode::Velocity;
    } else if (cmd.as<std::string>() == "trick") {
      c.entry.commands = qc::CommandMode::Trick;
    } else if (cmd.as<std::string>() == "false") {
      c.entry.commands = qc::CommandMode::Zero;
    } else {
      c.entry.commands = qc::CommandMode::Velocity;
    }
    catalog_.emplace_back(name, c);
  }

  using std::placeholders::_1;
  auto sub_text = [this](const char* topic, Command::Kind kind, bool lower) {
    subs_.push_back(create_subscription<std_msgs::msg::String>(topic, 10, [this, kind, lower](std_msgs::msg::String::SharedPtr m) {
      Command c = at(kind, steady_now());
      c.text = trim(m->data);
      if (lower) std::transform(c.text.begin(), c.text.end(), c.text.begin(), ::tolower);
      if (kind == Command::Kind::Mode) c.text = m->data;
      send(std::move(c));
    }));
  };
  auto sub_value = [this](const char* topic, Command::Kind kind) {
    subs_.push_back(create_subscription<std_msgs::msg::Float32>(topic, 10, [this, kind](std_msgs::msg::Float32::SharedPtr m) {
      Command c = at(kind, steady_now());
      c.value = m->data;
      send(std::move(c));
    }));
  };
  auto sub_values = [this](const char* topic, Command::Kind kind) {
    subs_.push_back(create_subscription<std_msgs::msg::Float32MultiArray>(
        topic, 10, [this, kind](std_msgs::msg::Float32MultiArray::SharedPtr m) {
          Command c = at(kind, steady_now());
          c.values.assign(m->data.begin(), m->data.end());
          send(std::move(c));
        }));
  };
  auto sub_flag = [this](const char* topic, Command::Kind kind, bool only_true) {
    subs_.push_back(create_subscription<std_msgs::msg::Bool>(topic, 10, [this, kind, only_true](std_msgs::msg::Bool::SharedPtr m) {
      if (only_true && !m->data) return;
      Command c = at(kind, steady_now());
      c.flag = m->data;
      send(std::move(c));
    }));
  };
  using K = Command::Kind;
  sub_text("/pipeline/policy", K::RequestPolicy, false);
  sub_text("/pipeline/mode", K::Mode, false);
  sub_text("/pose/command", K::PoseCommand, true);
  sub_value("/pose/interp_duration", K::PoseDuration);
  sub_values("/gait_command", K::Gait);
  sub_values("/gait_command/override", K::GaitOverride);
  sub_values("/pipeline/trick", K::Trick);
  subs_.push_back(create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr m) {
    Command c = at(K::CmdVel, steady_now());
    c.values = {m->linear.x, m->linear.y, m->angular.z};
    send(std::move(c));
  }));
  subs_.push_back(create_subscription<std_msgs::msg::Float32>("/safety/heartbeat", 10, [this](std_msgs::msg::Float32::SharedPtr) {
    const double now = steady_now();
    send(at(K::Heartbeat, now));
    std::lock_guard<std::mutex> lock(stats_mutex_);
    heartbeat_stats_.add(now);
  }));
  sub_value("/safety/max_torque_percent", K::MaxTorque);
  sub_value("/safety/base_tilt_limit_deg", K::TiltLimit);
  sub_value("/safety/base_forward_tilt_limit_deg", K::ForwardTiltLimit);
  sub_value("/safety/joint_rom_safety_margin", K::RomMargin);
  sub_value("/safety/watchdog_timeout", K::WatchdogTimeout);
  sub_flag("/safety/reset", K::SafetyReset, true);
  sub_flag("/safety/estop", K::Estop, true);
  sub_flag("/safety/numb", K::Numb, false);
  sub_value("/control/kp", K::Kp);
  sub_value("/control/kd", K::Kd);

  joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/sensors/joint_states", 10);
  imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/sensors/imu", 10);
  odom_sim_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom/state_simulator", 10);
  odom_est_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom/state_estimator", 10);
  gravity_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/estimator/projected_gravity", 10);
  lin_vel_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/estimator/base_lin_vel", 10);
  ang_vel_pub_ = create_publisher<geometry_msgs::msg::Vector3>("/estimator/base_ang_vel", 10);
  height_pub_ = create_publisher<std_msgs::msg::Float32>("/estimator/base_height", 10);
  contact_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/estimator/feet_contact", 10);
  leg_odom_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/estimator/leg_odometry", 10);
  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
  if (get<bool>(config_["odometry"], "publish_tf", true)) tf_pub_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  cmd_pub_ = create_publisher<sensor_msgs::msg::JointState>("/commands/joint_commands", 10);
  state_pub_ = create_publisher<std_msgs::msg::String>("/robot_state", 10);
  pose_status_pub_ = create_publisher<std_msgs::msg::String>("/pose/status", 10);
  estop_state_pub_ = create_publisher<std_msgs::msg::Bool>("/safety/estop_state", 10);
  foot_force_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/sensors/foot_force", 10);
  foot_cal_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/sensors/foot_force_calibration", 10);

  drain_timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] { drain_snapshots(); });
  calibration_timer_ = create_wall_timer(std::chrono::seconds(1), [this] { publish_calibration(); });
  light_timer_ = create_wall_timer(std::chrono::milliseconds(20), [this] { update_brightness(); });
  light_thread_ = std::thread(&DriverNode::light_worker, this);
  stdin_thread_ = std::thread(&DriverNode::stdin_loop, this);
  stdin_thread_.detach();
}

DriverNode::~DriverNode() {
  {
    std::lock_guard<std::mutex> lock(light_mutex_);
    light_stop_ = true;
  }
  light_cv_.notify_all();
  if (light_thread_.joinable()) light_thread_.join();
}

void DriverNode::light_worker() {
  // VUI calls are RPCs with a timeout: kept off the ROS thread and off the control loop.
  std::unique_lock<std::mutex> lock(light_mutex_);
  while (true) {
    light_cv_.wait(lock, [this] { return light_stop_ || light_request_; });
    if (light_stop_) return;
    const int level = *light_request_;
    light_request_.reset();
    lock.unlock();
    loop_.robot().set_brightness(level);
    lock.lock();
  }
}

Command DriverNode::at(Command::Kind kind, double now) {
  Command c;
  c.kind = kind;
  c.now = now;
  return c;
}

void DriverNode::send(Command c) {
  if (!commands_.push(std::move(c))) RCLCPP_ERROR(get_logger(), "[RealDriver] Command queue full; input dropped.");
}

// ---- policies ----

std::shared_ptr<qc::PolicyRunner> DriverNode::load_runner(const std::string& name, const std::string& checkpoint,
                                                          qc::PolicyEntry& entry) {
  if (checkpoint.empty() || !exists(checkpoint)) {
    RCLCPP_WARN(get_logger(), "[Policies] Policy file not found: %s", checkpoint.c_str());
    return nullptr;
  }
  RCLCPP_INFO(get_logger(), "[Policies] Loading policy '%s' from: %s", name.c_str(), checkpoint.c_str());
  const std::string base = checkpoint.substr(0, checkpoint.find_last_of('.'));
  const std::string qpol = base + ".qpol";

  // The checkpoint is read by quadruped_control's export.py once, into a .qpol beside it; reused while it is newer.
  auto do_export = [&]() -> bool {
    std::string script;
    try {
      script = ament_index_cpp::get_package_prefix("quadruped_control") + "/lib/quadruped_control/export_policy.py";
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "[Policies] quadruped_control not found: %s", e.what());
      return false;
    }
    RCLCPP_INFO(get_logger(), "[Policies] Exporting %s to %s ...", checkpoint.c_str(), qpol.c_str());
    std::vector<std::string> args = {"python3", script, checkpoint};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    pid_t pid;
    if (posix_spawnp(&pid, "python3", nullptr, nullptr, argv.data(), environ) != 0) return false;
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
  };

  try {
    if ((!exists(qpol) || mtime(qpol) < mtime(checkpoint)) && !do_export()) throw std::runtime_error("export failed");
    std::unique_ptr<qc::PolicySpec> spec;
    try {
      spec = std::make_unique<qc::PolicySpec>(qc::PolicySpec::from_file(qpol));
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "[Policies] %s unreadable (%s); exporting it again.", qpol.c_str(), e.what());
      if (!do_export()) throw std::runtime_error("export failed");
      spec = std::make_unique<qc::PolicySpec>(qc::PolicySpec::from_file(qpol));
    }
    if (entry.commands == qc::CommandMode::Trick) {
      entry.clock_hz = spec->trick_clock_hz;
      entry.switches = spec->trick_switches;
    }
    auto runner = std::make_shared<qc::PolicyRunner>(std::move(*spec));
    std::string extras;
    if (entry.commands == qc::CommandMode::Zero) extras += ", zero velocity command";
    if (entry.commands == qc::CommandMode::Trick) extras += ", trick command";
    if (entry.clock_hz) extras += ", " + json::num(entry.clock_hz) + " Hz clock";
    const auto prot = qc::protection_names(entry.protections);
    RCLCPP_INFO(get_logger(), "[Policies] '%s': protections %s%s.", name.c_str(),
                prot.empty() ? "none" : json::strings(prot).c_str(), extras.c_str());
    return runner;
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "[Policies] Failed to load policy '%s': %s", name.c_str(), e.what());
    return nullptr;
  }
}

void DriverNode::load_in_background(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(loading_mutex_);
    if (!loading_.insert(name).second) return;
  }
  std::thread([this, name] {
    auto it = std::find_if(catalog_.begin(), catalog_.end(), [&](const auto& c) { return c.first == name; });
    qc::PolicyEntry entry = it != catalog_.end() ? it->second.entry : qc::PolicyEntry{};
    auto runner = it != catalog_.end() ? load_runner(name, paths_.resolve(it->second.path), entry) : nullptr;
    Command c = at(runner ? Command::Kind::AddPolicy : Command::Kind::LoadError);
    c.text = name;
    c.runner = runner;
    c.entry = entry;
    c.detail = "could not load (see the driver's log)";
    send(std::move(c));
    std::lock_guard<std::mutex> lock(loading_mutex_);
    loading_.erase(name);
  }).detach();
}

std::unique_ptr<qc::Controller> DriverNode::make_controller(const std::string& checkpoint) {
  qc::ControllerConfig cfg = qc::ControllerConfig::from_yaml(paths_.config, paths_.geometry, kPolicyDt, true, true);
  // Every control step is a policy step: the motors run the PD between them.
  cfg.decimation = 1;
  cfg.policy_dt = kPolicyDt;
  auto ctl = std::make_unique<qc::Controller>(cfg);

  std::vector<std::pair<std::string, qc::PolicyEntry>> entries;
  for (const auto& [name, c] : catalog_) entries.emplace_back(name, c.entry);
  ctl->set_catalog(entries);

  if (!checkpoint.empty()) {
    const std::string ckpt = paths_.resolve(checkpoint);
    const std::string real = realpath_or(ckpt);
    std::optional<std::string> alias;
    qc::PolicyEntry entry;
    for (const auto& [name, c] : catalog_) {
      if (!c.path.empty() && realpath_or(paths_.resolve(c.path)) == real) {
        alias = name;
        entry = c.entry;
        break;
      }
    }
    if (auto runner = load_runner("main", ckpt, entry)) {
      ctl->add_policy("main", std::move(*runner), entry);
      ctl->set_main_alias(alias);
    }
  } else {
    RCLCPP_WARN(get_logger(), "[Pipeline] No main policy checkpoint provided. Policy runner disabled.");
  }

  std::string safety = safety_policy_;
  auto it = std::find_if(catalog_.begin(), catalog_.end(), [&](const auto& c) { return c.first == safety; });
  if (!safety.empty() && it == catalog_.end()) {
    RCLCPP_ERROR(get_logger(), "[Pipeline] safety_policy '%s' is not in policies.json.", safety.c_str());
    safety.clear();
  } else if (!safety.empty()) {
    qc::PolicyEntry entry = it->second.entry;
    if (auto runner = load_runner(safety, paths_.resolve(it->second.path), entry)) {
      ctl->add_policy(safety, std::move(*runner), entry);
    } else {
      safety.clear();
    }
  }
  ctl->set_safety_policy(safety);
  if (safety.empty()) {
    RCLCPP_WARN(get_logger(), "[Pipeline] No safety policy: the robot goes numb as soon as a check trips.");
  }
  ctl->set_active_policy(ctl->has_policy("main") ? std::optional<std::string>("main") : std::nullopt);
  for (const auto& e : ctl->drain_events()) RCLCPP_INFO(get_logger(), "%s", e.text.c_str());
  return ctl;
}

// ---- operator ----

void DriverNode::stdin_loop() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (estop_) {
      send(at(Command::Kind::ReleaseEstop));
    } else if (blocked_) {
      send(at(Command::Kind::ReleaseTrip));
      std::cout << kYellow << kBold << ">>> Safety RESET. Re-enabling main policy..." << kReset << std::endl;
      RCLCPP_INFO(get_logger(), "[Safety] Operator pressed ENTER. Main policy re-enabled.");
    }
  }
  stdin_ok_ = false;
}

void DriverNode::update_brightness() {
  const int want = loop_.desired_brightness();
  if (want < 0 || want == brightness_) return;
  brightness_ = want;
  {
    std::lock_guard<std::mutex> lock(light_mutex_);
    light_request_ = want;
  }
  light_cv_.notify_one();
}

// ---- snapshots ----

void DriverNode::drain_snapshots() {
  std::unique_ptr<Snapshot> s;
  while (snapshots_.pop(s)) {
    if (s) handle(*s);
  }
}

void DriverNode::handle(const Snapshot& s) {
  for (const auto& e : s.events) {
    if (e.kind == qc::EventKind::SafetyTrip) {
      print_shutdown_banner(e.text);
      if (!stdin_ok_) {
        RCLCPP_WARN(get_logger(), "[Safety] No interactive stdin; clear this stop from the console with 'safety = reset'.");
      }
    } else if (e.level == qc::LogLevel::Error) {
      RCLCPP_ERROR(get_logger(), "%s", e.text.c_str());
    } else if (e.level == qc::LogLevel::Warn) {
      RCLCPP_WARN(get_logger(), "%s", e.text.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "%s", e.text.c_str());
    }
  }
  estop_ = s.estop;
  blocked_ = s.blocked;
  if (s.estop != last_estop_) {
    std_msgs::msg::Bool b;
    b.data = s.estop;
    estop_state_pub_->publish(b);
    if (s.estop) {
      print_estop_banner();
      if (!stdin_ok_) {
        RCLCPP_ERROR(get_logger(), "[SAFETY] E-STOP latched, but this driver has no interactive stdin - nothing can release it. "
                                   "Restart the driver on the robot, or run it in a terminal.");
      }
    } else {
      char buf[96];
      std::snprintf(buf, sizeof(buf), ">>> E-STOP released. Torque ceiling back to %.1f Nm.", s.global_max_torque);
      std::cout << kYellow << kBold << buf << kReset << std::endl;
    }
    last_estop_ = s.estop;
  }
  for (const auto& name : s.load_requests) load_in_background(name);
  fsr_offset_ = s.fsr_offset;
  contact_threshold_ = s.contact_threshold;
  if (s.calibration_changed) publish_calibration();
  if (!s.has_state) return;

  if (s.pose_status) {
    std_msgs::msg::String m;
    m.data = *s.pose_status;
    pose_status_pub_->publish(m);
  }
  sensor_msgs::msg::JointState cmd;
  cmd.header.stamp = now();
  cmd.name.assign(joint_names_.begin(), joint_names_.end());
  cmd.position.assign(s.targets.data(), s.targets.data() + 12);
  cmd.effort.assign(12, s.max_torque);
  cmd_pub_->publish(cmd);
  if (s.foot_force) {
    std_msgs::msg::Float32MultiArray ff;
    ff.data.assign(s.foot_force->data(), s.foot_force->data() + 4);
    foot_force_pub_->publish(ff);
  }
  publish_telemetry(s);
  publish_robot_state(s);

  // A console started before the driver could hand a robot torque the moment it boots: refuse to run.
  if (!first_state_time_) first_state_time_ = s.sim_time;
  if (!console_checked_ && s.sim_time - *first_state_time_ >= 0.1) {
    console_checked_ = true;
    if (count_publishers("/safety/heartbeat") > 0) {
      RCLCPP_ERROR(get_logger(), "[Safety] Console was detected running before driver! Exiting driver for safety.");
      rclcpp::shutdown();
    }
  }
}

void DriverNode::publish_calibration() {
  if (!have_calibration_) return;
  std_msgs::msg::Float32MultiArray m;
  m.data = {static_cast<float>(contact_threshold_)};
  for (int i = 0; i < 4; ++i) m.data.push_back(static_cast<float>(fsr_offset_[i]));
  foot_cal_pub_->publish(m);
}

void DriverNode::publish_telemetry(const Snapshot& s) {
  const qc::State& st = s.state;
  const auto t = stamp(s.wall_time);
  const auto& q = st.quat_wxyz;

  sensor_msgs::msg::JointState js;
  js.header.stamp = t;
  js.name.assign(joint_names_.begin(), joint_names_.end());
  js.position.assign(st.q.data(), st.q.data() + 12);
  js.velocity.assign(st.dq.data(), st.dq.data() + 12);
  joint_pub_->publish(js);

  sensor_msgs::msg::Imu imu;
  imu.header.stamp = t;
  imu.header.frame_id = "imu_link";
  imu.orientation = quat(q[0], q[1], q[2], q[3]);
  imu.angular_velocity = vec3(st.gyro[0], st.gyro[1], st.gyro[2]);
  imu.linear_acceleration = vec3(st.accel[0], st.accel[1], st.accel[2]);
  imu_pub_->publish(imu);

  for (int k = 0; k < 2; ++k) {
    nav_msgs::msg::Odometry o;
    o.header.stamp = t;
    o.header.frame_id = "odom";
    o.child_frame_id = "base";
    o.pose.pose.position.x = st.base_pos[0];
    o.pose.pose.position.y = st.base_pos[1];
    o.pose.pose.position.z = st.base_pos[2];
    o.pose.pose.orientation = quat(q[0], q[1], q[2], q[3]);
    const qc::Vec3& v = k == 0 ? st.base_lin_vel_sim : st.base_lin_vel_est;
    o.twist.twist.linear = vec3(v[0], v[1], v[2]);
    (k == 0 ? odom_sim_pub_ : odom_est_pub_)->publish(o);
  }

  const qc::Vec3 g = qc::projected_gravity_b(q);
  gravity_pub_->publish(vec3(g[0], g[1], g[2]));
  lin_vel_pub_->publish(vec3(st.base_lin_vel[0], st.base_lin_vel[1], st.base_lin_vel[2]));
  ang_vel_pub_->publish(vec3(st.gyro[0], st.gyro[1], st.gyro[2]));
  std_msgs::msg::Float32 h;
  h.data = st.base_pos[2];
  height_pub_->publish(h);
  std_msgs::msg::Float32MultiArray fc;
  fc.data.assign(st.feet_contact.data(), st.feet_contact.data() + 4);
  contact_pub_->publish(fc);

  std_msgs::msg::Float32MultiArray lo;
  for (int leg = 0; leg < 4; ++leg) {
    const qc::Vec3 ql(st.q[leg], st.q[leg + 4], st.q[leg + 8]);
    const qc::Vec3 dql(st.dq[leg], st.dq[leg + 4], st.dq[leg + 8]);
    const qc::Vec3 v = -kinematics_.foot_jacobian_body(leg, ql) * dql - st.gyro.cross(kinematics_.foot_position_body(leg, ql));
    lo.data.push_back(static_cast<float>(v.norm()));
  }
  leg_odom_pub_->publish(lo);

  // Leg odometry: the estimated body velocity turned by the IMU orientation, integrated. The real robot has
  // no ground-truth position, so this is what odometry.source auto and leg both mean here.
  if (!odom_pos_) odom_pos_ = Eigen::Vector3d(0.0, 0.0, odom_height_);
  if (odom_time_ && s.sim_time - *odom_time_ > 0.0 && s.sim_time - *odom_time_ < 0.5) {
    const qc::Vec3 vw = qc::rot_from_quat(q) * st.base_lin_vel_est;
    odom_pos_->head<2>() += vw.head<2>() * (s.sim_time - *odom_time_);
  }
  odom_time_ = s.sim_time;
  const Eigen::Vector3d& pos = *odom_pos_;
  const double yaw = std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
  const auto footprint_rot = quat(std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2));
  if (tf_pub_) {
    geometry_msgs::msg::TransformStamped base, footprint;
    base.header.stamp = footprint.header.stamp = t;
    base.header.frame_id = footprint.header.frame_id = "odom";
    base.child_frame_id = "base";
    base.transform.translation = vec3(pos[0], pos[1], pos[2]);
    base.transform.rotation = quat(q[0], q[1], q[2], q[3]);
    footprint.child_frame_id = "base_footprint";
    footprint.transform.translation = vec3(pos[0], pos[1], 0.0);
    footprint.transform.rotation = footprint_rot;
    tf_pub_->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{base, footprint});
  }
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = t;
  odom.header.frame_id = "odom";
  odom.child_frame_id = "base_footprint";
  odom.pose.pose.position.x = pos[0];
  odom.pose.pose.position.y = pos[1];
  odom.pose.pose.orientation = footprint_rot;
  odom.twist.twist.linear = vec3(st.base_lin_vel[0], st.base_lin_vel[1], 0.0);
  odom.twist.twist.angular = vec3(0.0, 0.0, st.gyro[2]);
  odom_pub_->publish(odom);
}

void DriverNode::publish_robot_state(const Snapshot& s) {
  const double now = steady_now();
  if (now - state_time_ < state_period_) return;
  state_time_ = now;
  const qc::State& st = s.state;
  const qc::Vec3 g = qc::projected_gravity_b(st.quat_wxyz);
  const double tilt = std::acos(std::clamp(-g[2] / g.norm(), -1.0, 1.0)) * 180.0 / M_PI;
  const double vx = st.base_lin_vel[0], vy = st.base_lin_vel[1], yaw_rate = st.gyro[2];
  const std::vector<double> cmd = {r3(s.command[0]), r3(s.command[1]), r3(s.command[2])};
  const bool moving = std::max({std::abs(cmd[0]), std::abs(cmd[1]), std::abs(cmd[2])}) > 0.05 || std::hypot(vx, vy) > 0.15;
  std::string posture;
  if (tilt > 60.0) {
    posture = "fallen";
  } else if (s.max_torque <= 0.1 || (s.mode != "policy" && s.pose_name == "lie_flat")) {
    posture = "lying";
  } else if (s.mode == "policy") {
    posture = moving ? "walking" : "standing";
  } else {
    posture = s.pose_name;
  }

  std::string gait = "null";
  if (s.gait) {
    const Eigen::VectorXf& gv = *s.gait;
    json::Object o;
    for (int i = 0; i < gv.size() && i < 9; ++i) o.add(kGaitNames[i], json::num(r3(gv[i])));
    std::string best;
    float best_d = INFINITY;
    for (const auto& [name, preset] : kWtwGaits) {
      const float d = std::abs(preset[0] - gv[2]) + std::abs(preset[1] - gv[3]) + std::abs(preset[2] - gv[4]);
      if (d < best_d) best_d = d, best = name;
    }
    o.add("name", json::str(best)).add("override", json::boolean(s.gait_override));
    gait = o.dump();
  }
  auto opt = [](const std::optional<std::string>& v) { return v ? json::str(*v) : std::string("null"); };
  std::string heartbeat;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    heartbeat = heartbeat_stats_.summary(now);
  }
  std_msgs::msg::String m;
  m.data = json::Object()
               .add("t", json::num(r3(s.wall_time)))
               .add("mode", json::str(s.mode))
               .add("transition", json::boolean(s.transition))
               .add("posture", json::str(posture))
               .add("pose", json::str(s.pose_name))
               .add("pose_done", json::boolean(s.pose_done))
               .add("moving", json::boolean(moving))
               .add("estop", json::boolean(s.estop))
               .add("safety_blocked", json::boolean(s.blocked))
               .add("safety_policy_active", json::boolean(s.safety_policy_active))
               .add("trip_reason", json::str(s.trip_reason))
               .add("numb", json::boolean(s.numb))
               .add("numb_reason", json::str(s.numb_reason))
               .add("policy", opt(s.active))
               .add("policies", json::strings(s.available))
               .add("policy_loading", opt(s.pending))
               .add("policy_error", opt(s.error))
               .add("protections", json::strings(s.protections))
               .add("torque_on", json::boolean(s.max_torque > 0.1))
               .add("tilt_deg", json::num(std::round(tilt * 10.0) / 10.0))
               .add("velocity", json::floats({r3(vx), r3(vy), r3(yaw_rate)}))
               .add("cmd", json::floats(cmd))
               .add("trick", json::floats({s.trick[0], s.trick[1], s.trick[2]}))
               .add("trick_switches", json::integer(s.trick_switches))
               .add("gait", gait)
               .add("heartbeat", heartbeat)
               .dump();
  state_pub_->publish(m);
}

}  // namespace quadruped_real
