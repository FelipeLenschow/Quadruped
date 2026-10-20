// RoboSense Helios 16P to ROS 2: /helios/points for FAST-LIO.
//
// In: the lidar's UDP packets (MSOP, DIFOP), decoded by rs_driver (third_party/rs_driver). Or, with
// --from_topic <topic>, a recorded rslidar_sdk cloud (XYZIRT: x, y, z, intensity, ring, timestamp in s), to
// replay the old bags.
//
// Out: the cloud in the lidar frame (static TF base -> helios, config.yaml helios: xyz, rpy), in the Velodyne
// layout that FAST-LIO reads (lidar_type 2): x, y, z, intensity (float32), time (float32, s after the header
// stamp, the first point), ring (uint16). The point order stays the scan order. NaN points, points out of
// min_range .. max_range and points in body_box (base frame: the robot itself) are removed.
// The lidar stamps are this machine's clock (rs_driver use_lidar_clock off), the clock of /lidar/imu.
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <rs_driver/api/lidar_driver.hpp>
#include <rs_driver/msg/point_cloud_msg.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include "config_loader.hpp"
#include "sensor_mounts.hpp"

using namespace quadruped_real;
namespace rs = robosense::lidar;
using RsCloud = PointCloudT<PointXYZIRT>;

namespace {

constexpr double kStaleS = 1.0;
constexpr uint8_t kUint8 = 2, kUint16 = 4, kFloat32 = 7, kFloat64 = 8;

double monotonic() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::array<double, 3> triple(const YAML::Node& n, const std::array<double, 3>& fallback) {
  if (!n || !n.IsSequence() || n.size() != 3) return fallback;
  return {n[0].as<double>(), n[1].as<double>(), n[2].as<double>()};
}

// One point on its way out: t is its absolute time in s.
struct Point {
  float x, y, z, intensity;
  uint16_t ring;
  double t;
};

class HeliosDriver : public rclcpp::Node {
 public:
  HeliosDriver(const std::vector<std::string>& args, const YAML::Node& config) : Node("helios_driver") {
    const YAML::Node helios = config["helios"];
    frame_ = get<std::string>(helios, "frame", "helios");
    base_frame_ = arg(args, "--base_frame").empty() ? "base" : arg(args, "--base_frame");
    min_range_ = get<double>(helios, "min_range", 0.4);
    max_range_ = get<double>(helios, "max_range", 100.0);
    const Mount mount{triple(helios ? helios["xyz"] : YAML::Node(), {0.0, 0.0, 0.0}),
                      triple(helios ? helios["rpy"] : YAML::Node(), {0.0, 0.0, 0.0})};
    mount_r_ = matrix(mount.rpy);
    mount_t_ = Eigen::Vector3d(mount.xyz[0], mount.xyz[1], mount.xyz[2]);
    const YAML::Node box = helios ? helios["body_box"] : YAML::Node();
    if (box && box.IsMap()) {
      body_box_.emplace();
      const char* axes[3] = {"x", "y", "z"};
      for (int i = 0; i < 3; ++i) {
        (*body_box_)(i, 0) = box[axes[i]][0].as<double>();
        (*body_box_)(i, 1) = box[axes[i]][1].as<double>();
      }
    }

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/helios/points", rclcpp::SensorDataQoS());
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    tf_->sendTransform(transform(base_frame_, frame_, mount));
    RCLCPP_INFO(get_logger(), "mount xyz [%g, %g, %g] rpy [%g, %g, %g], range %.2f .. %.1f m%s", mount.xyz[0],
                mount.xyz[1], mount.xyz[2], mount.rpy[0], mount.rpy[1], mount.rpy[2], min_range_, max_range_,
                body_box_ ? ", body box on" : "");

    const std::string from_topic = arg(args, "--from_topic");
    if (!from_topic.empty()) {
      // Reliable: a recording plays every cloud, and a lost fragment of a large one loses all of it
      sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          from_topic, rclcpp::QoS(20).reliable(),
          [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_recorded(*m); });
      RCLCPP_INFO(get_logger(), "%s (rslidar_sdk XYZIRT) -> /helios/points", from_topic.c_str());
    } else {
      start_lidar(helios);
    }
    watch_timer_ = create_wall_timer(std::chrono::milliseconds(200), [this] { watch(); });
  }

  void close() {
    stop_ = true;
    cloud_ready_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (driver_) driver_->stop();
  }

 private:
  void start_lidar(const YAML::Node& helios) {
    rs::RSDriverParam param;
    param.input_type = rs::InputType::ONLINE_LIDAR;
    param.lidar_type = rs::LidarType::RSHELIOS_16P;
    param.input_param.msop_port = get<int>(helios, "msop_port", 6699);
    param.input_param.difop_port = get<int>(helios, "difop_port", 7788);
    param.input_param.host_address = get<std::string>(helios, "host_address", "0.0.0.0");
    param.input_param.group_address = get<std::string>(helios, "group_address", "0.0.0.0");
    param.decoder_param.use_lidar_clock = false;   // this machine's clock, as /lidar/imu
    param.decoder_param.dense_points = true;       // no NaN points
    param.decoder_param.wait_for_difop = true;     // the angle calibration first

    driver_ = std::make_unique<rs::LidarDriver<RsCloud>>();
    // Two clouds take turns: the driver fills one while the worker publishes the other
    driver_->regPointCloudCallback(
        [this]() {
          std::lock_guard<std::mutex> lock(cloud_mutex_);
          if (free_) return std::exchange(free_, nullptr);
          return std::make_shared<RsCloud>();
        },
        [this](std::shared_ptr<RsCloud> cloud) {
          {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            if (ready_) dropped_++;   // the worker is behind: the older cloud goes
            ready_ = std::move(cloud);
          }
          cloud_ready_.notify_one();
        });
    driver_->regExceptionCallback([this](const rs::Error& e) {
      // Packet loss and timeouts repeat: one line per kind
      std::lock_guard<std::mutex> lock(error_mutex_);
      if (errors_seen_.insert(static_cast<int>(e.error_code)).second) RCLCPP_WARN(get_logger(), "rs_driver: %s", e.toString().c_str());
    });
    if (!driver_->init(param)) {
      RCLCPP_ERROR(get_logger(), "rs_driver init failed (ports %d, %d in use?)", param.input_param.msop_port,
                   param.input_param.difop_port);
      throw std::runtime_error("rs_driver init");
    }
    worker_ = std::thread([this] { work(); });
    driver_->start();
    RCLCPP_INFO(get_logger(), "Helios 16P on UDP %d (MSOP), %d (DIFOP) -> /helios/points", param.input_param.msop_port,
                param.input_param.difop_port);
  }

  void work() {
    std::vector<Point> points;
    while (!stop_) {
      std::shared_ptr<RsCloud> cloud;
      {
        std::unique_lock<std::mutex> lock(cloud_mutex_);
        cloud_ready_.wait(lock, [this] { return stop_ || ready_; });
        if (stop_) return;
        cloud = std::exchange(ready_, nullptr);
      }
      points.clear();
      points.reserve(cloud->points.size());
      for (const auto& p : cloud->points)
        points.push_back({p.x, p.y, p.z, static_cast<float>(p.intensity), p.ring, p.timestamp});
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        free_ = std::move(cloud);
      }
      publish(points);
    }
  }

  void on_recorded(const sensor_msgs::msg::PointCloud2& m) {
    struct Field {
      int offset = -1;
      uint8_t type = 0;
    };
    Field x, y, z, intensity, ring, timestamp;
    for (const auto& f : m.fields) {
      Field* field = f.name == "x"           ? &x
                     : f.name == "y"         ? &y
                     : f.name == "z"         ? &z
                     : f.name == "intensity" ? &intensity
                     : f.name == "ring"      ? &ring
                     : f.name == "timestamp" ? &timestamp
                                             : nullptr;
      if (field) *field = {static_cast<int>(f.offset), f.datatype};
    }
    if (m.is_bigendian || x.type != kFloat32 || y.type != kFloat32 || z.type != kFloat32 || ring.type != kUint16 ||
        timestamp.type != kFloat64 || (intensity.type != kFloat32 && intensity.type != kUint8)) {
      if (!warned_format_) {
        warned_format_ = true;
        RCLCPP_ERROR(get_logger(), "the recorded cloud is not rslidar_sdk XYZIRT (float32 x y z, uint16 ring, "
                                   "float64 timestamp): not used");
      }
      return;
    }
    recorded_.clear();
    const size_t n = size_t(m.width) * m.height;
    recorded_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const uint8_t* p = m.data.data() + i * m.point_step;
      Point q;
      std::memcpy(&q.x, p + x.offset, 4);
      std::memcpy(&q.y, p + y.offset, 4);
      std::memcpy(&q.z, p + z.offset, 4);
      if (intensity.type == kFloat32) {
        std::memcpy(&q.intensity, p + intensity.offset, 4);
      } else {
        q.intensity = p[intensity.offset];
      }
      std::memcpy(&q.ring, p + ring.offset, 2);
      std::memcpy(&q.t, p + timestamp.offset, 8);
      recorded_.push_back(q);
    }
    publish(recorded_);
  }

  // The kept points in scan order, time after the first one
  void publish(const std::vector<Point>& points) {
    std::optional<double> t0;
    size_t kept = 0;
    out_.data.resize(points.size() * kPointStep);
    uint8_t* d = out_.data.data();
    for (const auto& p : points) {
      if (!keep(p)) continue;
      if (!t0) t0 = p.t;
      const float time = static_cast<float>(p.t - *t0);
      uint8_t* q = d + kept * kPointStep;
      std::memcpy(q + 0, &p.x, 4);
      std::memcpy(q + 4, &p.y, 4);
      std::memcpy(q + 8, &p.z, 4);
      std::memcpy(q + 12, &p.intensity, 4);
      std::memcpy(q + 16, &time, 4);
      std::memcpy(q + 20, &p.ring, 2);
      std::memset(q + 22, 0, 2);
      ++kept;
    }
    if (!t0) return;
    out_.data.resize(kept * kPointStep);
    out_.header.stamp = rclcpp::Time(static_cast<int64_t>(std::llround(*t0 * 1e9)));
    out_.header.frame_id = frame_;
    out_.height = 1;
    out_.width = static_cast<uint32_t>(kept);
    out_.is_bigendian = false;
    out_.is_dense = true;
    out_.point_step = kPointStep;
    out_.row_step = static_cast<uint32_t>(kept * kPointStep);
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
    {
      std::lock_guard<std::mutex> lock(watch_mutex_);
      if (last_cloud_ < 0.0) {
        const double span = points.empty() ? 0.0 : points.back().t - *t0;
        RCLCPP_INFO(get_logger(), "first cloud: %zu of %zu points kept, time 0 .. %.4f s", kept, points.size(), span);
      } else if (stale_) {
        RCLCPP_INFO(get_logger(), "lidar cloud resumed");
      }
      last_cloud_ = monotonic();
      stale_ = false;
    }
    pub_->publish(out_);
  }

  bool keep(const Point& p) const {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;
    const double r2 = double(p.x) * p.x + double(p.y) * p.y + double(p.z) * p.z;
    if (r2 < min_range_ * min_range_ || r2 > max_range_ * max_range_) return false;
    if (body_box_) {
      const Eigen::Vector3d b = mount_r_ * Eigen::Vector3d(p.x, p.y, p.z) + mount_t_;
      bool inside = true;
      for (int c = 0; c < 3; ++c) inside = inside && b[c] >= (*body_box_)(c, 0) && b[c] <= (*body_box_)(c, 1);
      if (inside) return false;
    }
    return true;
  }

  void watch() {
    std::lock_guard<std::mutex> lock(watch_mutex_);
    const double now = monotonic();
    if (last_cloud_ >= 0.0 && !stale_ && now - last_cloud_ > kStaleS) {
      stale_ = true;
      RCLCPP_WARN(get_logger(), "no lidar cloud for %.0f s", kStaleS);
    }
    if (last_cloud_ < 0.0 && !warned_none_ && now - started_ > 5.0) {
      warned_none_ = true;
      RCLCPP_WARN(get_logger(), "no cloud after 5 s: is the lidar on, and this machine on 192.168.1.102?");
    }
    const unsigned long dropped = dropped_.exchange(0);
    if (dropped > 0) RCLCPP_WARN(get_logger(), "%lu clouds dropped: publishing is slower than the lidar", dropped);
  }

  static constexpr uint32_t kPointStep = 24;

  std::string frame_, base_frame_;
  double min_range_, max_range_;
  Eigen::Matrix3d mount_r_;
  Eigen::Vector3d mount_t_;
  std::optional<Eigen::Matrix<double, 3, 2>> body_box_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::TimerBase::SharedPtr watch_timer_;
  sensor_msgs::msg::PointCloud2 out_;
  std::vector<Point> recorded_;

  std::unique_ptr<rs::LidarDriver<RsCloud>> driver_;
  std::thread worker_;
  std::mutex cloud_mutex_;
  std::condition_variable cloud_ready_;
  std::shared_ptr<RsCloud> free_, ready_;
  std::atomic<bool> stop_{false};
  std::atomic<unsigned long> dropped_{0};
  std::mutex error_mutex_;
  std::set<int> errors_seen_;

  std::mutex watch_mutex_;
  double started_ = monotonic(), last_cloud_ = -1.0;
  bool stale_ = false, warned_none_ = false, warned_format_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const Paths paths = Paths::find();
  const YAML::Node config = load_config(paths, "helios_driver");
  use_ros_domain(config);
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    auto node = std::make_shared<HeliosDriver>(args, config);
    rclcpp::spin(node);
    node->close();
  } catch (const std::exception& e) {
    std::cerr << "[helios_driver] " << e.what() << std::endl;
    code = 1;
  }
  if (rclcpp::ok()) rclcpp::shutdown();
  return code;
}
