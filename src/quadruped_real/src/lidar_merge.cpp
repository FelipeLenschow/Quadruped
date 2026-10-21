// The Helios 16P and the Go2's L1 as one lidar for FAST-LIO: /lidars/points.
//
// In: /helios/points (helios_driver) and /lidar/points (real_sensors), each in its own frame, with a per-point
// time (float32, s after the header stamp). Both are on this machine's clock, the clock of /lidar/imu.
//
// Out: /lidars/points in the base frame (FAST-LIO's extrinsic is then the identity), in the Velodyne layout
// helios_driver writes: x, y, z, intensity (float32), time (float32, s after the header stamp), ring (uint16;
// the L1's rings follow the Helios' 16). The points are sorted by time.
//
// The Helios sets the pace: each of its scans is one merged scan, with the L1 points of the same time span. A
// scan covers from the end of the previous one to the end of its Helios scan, so no L1 point falls between two;
// it goes out once the L1 has passed its end, or after --wait_ms. Without the L1 the Helios goes out alone.
// --helios_every / --l1_every keep every n-th point of each (FAST-LIO's point_filter_num is then 1): the L1
// has ~6x fewer points, and a shared filter would thin it as much as the Helios.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <optional>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "config_loader.hpp"
#include "sensor_mounts.hpp"

using namespace quadruped_real;

namespace {

constexpr uint8_t kUint16 = 4, kFloat32 = 7;
constexpr uint32_t kPointStep = 24;
constexpr uint16_t kL1RingOffset = 16;
constexpr double kMaxSpanS = 0.25;   // a merged scan never starts earlier than this before its Helios scan
constexpr double kL1StaleS = 0.5;    // no L1 cloud for this long: the Helios goes out without waiting
constexpr double kReportS = 10.0;

std::array<double, 3> triple(const YAML::Node& n, const std::array<double, 3>& fallback) {
  if (!n || !n.IsSequence() || n.size() != 3) return fallback;
  return {n[0].as<double>(), n[1].as<double>(), n[2].as<double>()};
}

double wall() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double seconds(const builtin_interfaces::msg::Time& t) { return t.sec + t.nanosec * 1e-9; }

// One point in the base frame; t is its absolute time in s.
struct Point {
  float x, y, z, intensity;
  uint16_t ring;
  double t;
};

struct Source {
  std::string name;
  Eigen::Matrix3f r;
  Eigen::Vector3f t;
  int every = 1;
  uint16_t ring_offset = 0;
  bool warned_format = false;
};

// The points of a cloud with x, y, z, intensity, time (float32) and ring (uint16), in the base frame. Non-finite
// points (the L1's blanked ones) are skipped. False if the layout is not that.
bool read(const sensor_msgs::msg::PointCloud2& m, Source& src, std::vector<Point>& out) {
  int x = -1, y = -1, z = -1, intensity = -1, time = -1, ring = -1;
  bool types_ok = true;
  for (const auto& f : m.fields) {
    int* offset = f.name == "x"           ? &x
                  : f.name == "y"         ? &y
                  : f.name == "z"         ? &z
                  : f.name == "intensity" ? &intensity
                  : f.name == "time"      ? &time
                  : f.name == "ring"      ? &ring
                                          : nullptr;
    if (!offset) continue;
    *offset = static_cast<int>(f.offset);
    types_ok = types_ok && f.datatype == (f.name == "ring" ? kUint16 : kFloat32);
  }
  if (m.is_bigendian || !types_ok || x < 0 || y < 0 || z < 0 || intensity < 0 || time < 0 || ring < 0) return false;
  const double stamp = seconds(m.header.stamp);
  const size_t n = size_t(m.width) * m.height;
  out.reserve(out.size() + n / src.every + 1);
  for (size_t i = 0; i < n; i += src.every) {
    const uint8_t* p = m.data.data() + i * m.point_step;
    float v[3], in, dt;
    uint16_t r;
    std::memcpy(&v[0], p + x, 4);
    std::memcpy(&v[1], p + y, 4);
    std::memcpy(&v[2], p + z, 4);
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) continue;
    std::memcpy(&in, p + intensity, 4);
    std::memcpy(&dt, p + time, 4);
    std::memcpy(&r, p + ring, 2);
    const Eigen::Vector3f b = src.r * Eigen::Vector3f(v[0], v[1], v[2]) + src.t;
    out.push_back({b.x(), b.y(), b.z(), in, static_cast<uint16_t>(r + src.ring_offset), stamp + dt});
  }
  return true;
}

class LidarMerge : public rclcpp::Node {
 public:
  LidarMerge(const std::vector<std::string>& args, const YAML::Node& config) : Node("lidar_merge") {
    base_frame_ = arg(args, "--base_frame").empty() ? "base" : arg(args, "--base_frame");
    const std::string every_h = arg(args, "--helios_every"), every_l = arg(args, "--l1_every"),
                      wait = arg(args, "--wait_ms");
    wait_s_ = (wait.empty() ? 80.0 : std::stod(wait)) * 1e-3;   // an L1 cloud ends up to ~65 ms after a Helios scan

    const YAML::Node helios = config["helios"], l1 = config["real_lidar"];
    helios_ = source("helios", helios, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}, every_h.empty() ? 3 : std::stoi(every_h), 0);
    l1_ = source("L1", l1, kRadarMount, every_l.empty() ? 1 : std::stoi(every_l), kL1RingOffset);

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/lidars/points", rclcpp::SensorDataQoS());
    helios_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/helios/points", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_helios(*m); });
    l1_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/lidar/points", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_l1(*m); });
    timer_ = create_wall_timer(std::chrono::milliseconds(5), [this] { flush(false); });
    RCLCPP_INFO(get_logger(), "/helios/points (every %d) + /lidar/points (every %d) -> /lidars/points in %s, "
                "waits %.0f ms for the L1", helios_.every, l1_.every, base_frame_.c_str(), wait_s_ * 1e3);
  }

 private:
  Source source(const std::string& name, const YAML::Node& n, const Mount& fallback, int every, uint16_t ring_offset) {
    const Mount mount{triple(n ? n["xyz"] : YAML::Node(), fallback.xyz), triple(n ? n["rpy"] : YAML::Node(), fallback.rpy)};
    Source s;
    s.name = name;
    s.r = matrix(mount.rpy).cast<float>();
    s.t = Eigen::Vector3f(mount.xyz[0], mount.xyz[1], mount.xyz[2]);
    s.every = std::max(1, every);
    s.ring_offset = ring_offset;
    RCLCPP_INFO(get_logger(), "%s mount xyz [%g, %g, %g] rpy [%g, %g, %g]", name.c_str(), mount.xyz[0], mount.xyz[1],
                mount.xyz[2], mount.rpy[0], mount.rpy[1], mount.rpy[2]);
    return s;
  }

  bool parse(const sensor_msgs::msg::PointCloud2& m, Source& src, std::vector<Point>& out) {
    if (read(m, src, out)) return true;
    if (!src.warned_format) {
      src.warned_format = true;
      RCLCPP_ERROR(get_logger(), "%s cloud has no float32 x y z intensity time and uint16 ring: not used",
                   src.name.c_str());
    }
    return false;
  }

  void on_helios(const sensor_msgs::msg::PointCloud2& m) {
    if (pending_) flush(true);   // the next scan is here: the previous one goes as it is
    std::vector<Point> points;
    if (!parse(m, helios_, points) || points.empty()) return;
    double end = seconds(m.header.stamp);
    for (const auto& p : points) end = std::max(end, p.t);
    pending_ = Pending{std::move(points), seconds(m.header.stamp), end, wall()};
    ++stats_.helios;
    flush(false);
  }

  void on_l1(const sensor_msgs::msg::PointCloud2& m) {
    std::vector<Point> points;
    if (!parse(m, l1_, points)) return;
    // In time order, after the previous clouds: flush takes from the front up to a scan's end
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.t < b.t; });
    for (const auto& p : points) {
      if (p.t < published_end_) {
        ++stats_.l1_late;   // its scan has gone already
        continue;
      }
      l1_buffer_.push_back(p);
      l1_seen_ = std::max(l1_seen_, p.t);
    }
    l1_wall_ = wall();
    ++stats_.l1;
    flush(false);
  }

  // The pending Helios scan goes out once the L1 has passed its end, after wait_s_, or when forced.
  void flush(bool force) {
    report();
    if (!pending_) return;
    const double now = wall();
    const bool l1_live = l1_wall_ > 0.0 && now - l1_wall_ < kL1StaleS;
    if (!force && l1_live && l1_seen_ < pending_->end && now - pending_->arrived < wait_s_) return;
    if (l1_live && l1_seen_ < pending_->end) ++stats_.waited_out;

    // From the end of the previous scan (or the Helios scan's start, if that was long ago) to this one's end
    const double start = std::max(published_end_ > 0.0 ? published_end_ : pending_->start, pending_->start - kMaxSpanS);
    std::vector<Point>& points = pending_->points;
    size_t l1_used = 0;
    while (!l1_buffer_.empty() && l1_buffer_.front().t <= pending_->end) {
      const Point& p = l1_buffer_.front();
      if (p.t >= start) {
        points.push_back(p);
        ++l1_used;
      }
      l1_buffer_.pop_front();
    }
    // FAST-LIO reads the last point's time to tell that the cloud has per-point times
    std::sort(points.begin(), points.end(), [](const Point& a, const Point& b) { return a.t < b.t; });
    stats_.l1_points += l1_used;
    publish(points, std::min(start, points.front().t));
    published_end_ = pending_->end;
    pending_.reset();
  }

  void publish(const std::vector<Point>& points, double stamp) {
    out_.data.resize(points.size() * kPointStep);
    uint8_t* d = out_.data.data();
    for (size_t i = 0; i < points.size(); ++i) {
      const Point& p = points[i];
      const float time = static_cast<float>(p.t - stamp);
      uint8_t* q = d + i * kPointStep;
      std::memcpy(q + 0, &p.x, 4);
      std::memcpy(q + 4, &p.y, 4);
      std::memcpy(q + 8, &p.z, 4);
      std::memcpy(q + 12, &p.intensity, 4);
      std::memcpy(q + 16, &time, 4);
      std::memcpy(q + 20, &p.ring, 2);
      std::memset(q + 22, 0, 2);
    }
    out_.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(stamp * 1e9)));
    out_.header.frame_id = base_frame_;
    out_.height = 1;
    out_.width = static_cast<uint32_t>(points.size());
    out_.is_bigendian = false;
    out_.is_dense = true;
    out_.point_step = kPointStep;
    out_.row_step = static_cast<uint32_t>(points.size() * kPointStep);
    if (out_.fields.empty()) {
      const std::pair<const char*, std::pair<uint32_t, uint8_t>> fields[] = {
          {"x", {0, kFloat32}}, {"y", {4, kFloat32}},     {"z", {8, kFloat32}},
          {"intensity", {12, kFloat32}}, {"time", {16, kFloat32}}, {"ring", {20, kUint16}}};
      for (const auto& [name, layout] : fields) {
        sensor_msgs::msg::PointField f;
        f.name = name;
        f.offset = layout.first;
        f.datatype = layout.second;
        f.count = 1;
        out_.fields.push_back(f);
      }
    }
    pub_->publish(out_);
    ++stats_.out;
    stats_.out_points += points.size();
  }

  void report() {
    const double now = wall();
    if (now - stats_.since < kReportS) return;
    const double s = now - stats_.since;
    if (stats_.helios > 0 || stats_.l1 > 0) {
      RCLCPP_INFO(get_logger(), "%.1f Hz out (%.0f points: %.0f from the L1), Helios %.1f Hz, L1 %.1f Hz; "
                  "%lu scans did not wait for the L1, %lu late L1 points",
                  stats_.out / s, stats_.out ? double(stats_.out_points) / stats_.out : 0.0,
                  stats_.out ? double(stats_.l1_points) / stats_.out : 0.0, stats_.helios / s, stats_.l1 / s,
                  stats_.waited_out, stats_.l1_late);
      if (stats_.l1 == 0) RCLCPP_WARN(get_logger(), "no /lidar/points: the Helios alone (is real_sensors running?)");
    }
    stats_ = Stats{};
    stats_.since = now;
  }

  struct Pending {
    std::vector<Point> points;
    double start, end, arrived;
  };
  struct Stats {
    double since = wall();
    unsigned long helios = 0, l1 = 0, out = 0, out_points = 0, l1_points = 0, waited_out = 0, l1_late = 0;
  };

  std::string base_frame_;
  double wait_s_;
  Source helios_, l1_;
  std::optional<Pending> pending_;
  std::deque<Point> l1_buffer_;
  double l1_seen_ = 0.0, l1_wall_ = 0.0, published_end_ = 0.0;
  Stats stats_;
  sensor_msgs::msg::PointCloud2 out_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr helios_sub_, l1_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const Paths paths = Paths::find();
  const YAML::Node config = load_config(paths, "lidar_merge");
  use_ros_domain(config);
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    rclcpp::spin(std::make_shared<LidarMerge>(args, config));
  } catch (const std::exception& e) {
    std::cerr << "[lidar_merge] " << e.what() << std::endl;
    code = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return code;
}
