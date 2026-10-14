// Real Go2 sensors to ROS 2: the L1 lidar cloud and IMU from Unitree's DDS, on this machine's clock.
//
// Both keep the robot's own stamps, shifted by one offset onto this clock (RobotClock), so Point-LIO sees the
// cloud and IMU on the same timeline. /lidar/imu is the L1's gyro with the body IMU's accelerometer (the L1's
// own is not real), both in the body's axes.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/ros2/Imu_.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "config_loader.hpp"
#include "sensor_mounts.hpp"

using namespace quadruped_real;
using DdsCloud = sensor_msgs::msg::dds_::PointCloud2_;
using DdsImu = sensor_msgs::msg::dds_::Imu_;
using DdsString = std_msgs::msg::dds_::String_;
using DdsHeader = std_msgs::msg::dds_::Header_;
using LowState = unitree_go::msg::dds_::LowState_;

namespace {

const char* kLidarSwitchTopic = "rt/utlidar/switch";
const char* kLowStateTopic = "rt/lowstate";
constexpr double kStaleS = 1.0;
constexpr double kMinRange = 0.05;  // the L1's own minimum range; nearer after the range correction is not a real return
constexpr double kClockWindowS = 10.0;
constexpr uint8_t kFloat32 = 7, kFloat64 = 8;

double monotonic() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Maps the robot's stamps onto this clock: offset = the smallest (arrival - stamp) seen in the last
// kClockWindowS, i.e. the stamp plus the least transport delay. The window follows slow clock drift.
class RobotClock {
 public:
  explicit RobotClock(rclcpp::Clock::SharedPtr clock) : clock_(std::move(clock)) {}

  builtin_interfaces::msg::Time stamp(const DdsHeader& header, const std::string& stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    const rclcpp::Time now = clock_->now();
    const int64_t robot_ns = int64_t(header.stamp().sec()) * 1000000000LL + header.stamp().nanosec();
    if (robot_ns == 0) return now;
    const double t = now.nanoseconds() * 1e-9;
    const int64_t delta = now.nanoseconds() - robot_ns;
    auto it = least_.find(stream);
    if (it == least_.end() || delta < it->second) least_[stream] = delta;
    while (!deltas_.empty() && deltas_.back().second >= delta) deltas_.pop_back();
    deltas_.emplace_back(t, delta);
    while (deltas_.front().first < t - kClockWindowS) deltas_.pop_front();
    return rclcpp::Time(robot_ns + deltas_.front().second);
  }

  // The least (arrival - stamp) per stream since the last call, in ns.
  std::map<std::string, int64_t> take_least() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::exchange(least_, {});
  }

 private:
  rclcpp::Clock::SharedPtr clock_;
  std::mutex mutex_;
  std::deque<std::pair<double, int64_t>> deltas_;
  std::map<std::string, int64_t> least_;
};

std::array<double, 3> triple(const YAML::Node& n, const std::array<double, 3>& fallback) {
  if (!n || !n.IsSequence() || n.size() != 3) return fallback;
  return {n[0].as<double>(), n[1].as<double>(), n[2].as<double>()};
}

std::string list(const std::array<double, 3>& v) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "[%g, %g, %g]", v[0], v[1], v[2]);
  return buf;
}

std::optional<std::pair<const sensor_msgs::msg::dds_::PointField_*, uint32_t>> field(const DdsCloud& m,
                                                                                     const std::string& name) {
  for (const auto& f : m.fields())
    if (f.name() == name) return std::make_pair(&f, f.offset());
  return std::nullopt;
}

class RealSensors : public rclcpp::Node {
 public:
  RealSensors(const std::vector<std::string>& args, const YAML::Node& config)
      : Node("real_sensors"), clock_(get_clock()) {
    const std::string topic = arg(args, "--topic").empty() ? "rt/utlidar/cloud" : arg(args, "--topic");
    const std::string imu_topic = arg(args, "--imu_topic").empty() ? "rt/utlidar/imu" : arg(args, "--imu_topic");
    frame_ = arg(args, "--frame").empty() ? "radar" : arg(args, "--frame");
    base_frame_ = arg(args, "--base_frame").empty() ? "base" : arg(args, "--base_frame");

    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/lidar/points", 5);
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/lidar/imu", 200);

    const YAML::Node lidar = config["real_lidar"];
    const Mount mount{triple(lidar ? lidar["xyz"] : YAML::Node(), kRadarMount.xyz),
                      triple(lidar ? lidar["rpy"] : YAML::Node(), kRadarMount.rpy)};
    range_offset_ = get<double>(lidar, "range_offset", 0.0);
    const YAML::Node box = lidar ? lidar["body_box"] : YAML::Node();
    if (box && box.IsMap()) {
      body_box_.emplace();
      const char* axes[3] = {"x", "y", "z"};
      for (int i = 0; i < 3; ++i) {
        (*body_box_)(i, 0) = box[axes[i]][0].as<double>();
        (*body_box_)(i, 1) = box[axes[i]][1].as<double>();
      }
    }
    mount_t_ = Eigen::Vector3d(mount.xyz[0], mount.xyz[1], mount.xyz[2]);
    // The L1's gyro reads in the cloud's axes; turned into the body's, Point-LIO's world starts upright.
    mount_r_ = matrix(mount.rpy);
    tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    tf_->sendTransform(transform(base_frame_, "radar", mount));
    RCLCPP_INFO(get_logger(), "lidar mount xyz %s rpy %s, range offset %.3f m", list(mount.xyz).c_str(),
                list(mount.rpy).c_str(), range_offset_);

    if (!flag(args, "--no_switch_on")) {
      switch_.reset(new unitree::robot::ChannelPublisher<DdsString>(kLidarSwitchTopic));
      switch_->InitChannel();
    }
    watch_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { watch(); });
    report_timer_ = create_wall_timer(std::chrono::seconds(5), [this] { report(); });

    // The newest LowState only: 500 Hz, and 50 Hz is plenty for gravity.
    lowstate_.reset(new unitree::robot::ChannelSubscriber<LowState>(kLowStateTopic));
    lowstate_->InitChannel([this](const void* m) { on_lowstate(*static_cast<const LowState*>(m)); }, 1);
    imu_.reset(new unitree::robot::ChannelSubscriber<DdsImu>(imu_topic));
    imu_->InitChannel([this](const void* m) { on_imu(*static_cast<const DdsImu*>(m)); }, 100);
    cloud_.reset(new unitree::robot::ChannelSubscriber<DdsCloud>(topic));
    cloud_->InitChannel([this](const void* m) { on_cloud(*static_cast<const DdsCloud*>(m)); }, 1);
    RCLCPP_INFO(get_logger(), "%s -> /lidar/points (%s), %s -> /lidar/imu", topic.c_str(), frame_.c_str(),
                imu_topic.c_str());
  }

  void close() {
    if (cloud_) cloud_->CloseChannel();
    if (imu_) imu_->CloseChannel();
    if (lowstate_) lowstate_->CloseChannel();
    cloud_.reset();
    imu_.reset();
    lowstate_.reset();
    switch_.reset();
  }

 private:
  void on_lowstate(const LowState& s) {
    const auto& a = s.imu_state().accelerometer();
    std::lock_guard<std::mutex> lock(acc_mutex_);
    body_acc_ = Eigen::Vector3d(a[0], a[1], a[2]);
  }

  void on_imu(const DdsImu& m) {
    std::optional<Eigen::Vector3d> acc;
    {
      std::lock_guard<std::mutex> lock(acc_mutex_);
      acc = body_acc_;
    }
    if (!acc) return;
    ++imu_count_;
    sensor_msgs::msg::Imu msg;
    msg.header.stamp = clock_.stamp(m.header(), "imu");
    msg.header.frame_id = base_frame_;
    msg.orientation_covariance[0] = -1.0;
    const auto& w = m.angular_velocity();
    const Eigen::Vector3d gyro = mount_r_ * Eigen::Vector3d(w.x(), w.y(), w.z());
    msg.angular_velocity.x = gyro.x();
    msg.angular_velocity.y = gyro.y();
    msg.angular_velocity.z = gyro.z();
    msg.linear_acceleration.x = acc->x();
    msg.linear_acceleration.y = acc->y();
    msg.linear_acceleration.z = acc->z();
    imu_pub_->publish(msg);
  }

  void on_cloud(const DdsCloud& m) {
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      if (last_cloud_ < 0.0) {
        std::string names;
        for (const auto& f : m.fields()) names += (names.empty() ? "" : ", ") + f.name();
        RCLCPP_INFO(get_logger(), "first cloud: frame '%s', %u points, fields [%s]%s",
                    m.header().frame_id().c_str(), m.width() * m.height(), names.c_str(), time_range(m).c_str());
      } else if (stale_) {
        RCLCPP_INFO(get_logger(), "lidar cloud resumed");
      }
      last_cloud_ = monotonic();
      stale_ = false;
    }
    publish(m);
  }

  void watch() {
    const double now = monotonic();
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    if (last_cloud_ >= 0.0 && !stale_ && now - last_cloud_ > kStaleS) {
      stale_ = true;
      RCLCPP_WARN(get_logger(), "no lidar cloud for %.0f s", kStaleS);
    }
    if (switch_ && last_cloud_ < 0.0 && now - last_switch_ > 1.0) {
      last_switch_ = now;
      DdsString on;
      on.data("ON");
      switch_->Write(on);
    }
  }

  void report() {
    const long count = imu_count_.exchange(0);
    bool have_acc;
    {
      std::lock_guard<std::mutex> lock(acc_mutex_);
      have_acc = body_acc_.has_value();
    }
    if (count == 0 && !warned_imu_) {
      warned_imu_ = true;
      RCLCPP_WARN(get_logger(), "no lidar IMU samples (Point-LIO needs them)");
    }
    if (!have_acc) {
      if (!warned_acc_) {
        warned_acc_ = true;
        RCLCPP_WARN(get_logger(), "no %s: /lidar/imu waits for the body accelerometer", kLowStateTopic);
      }
    } else if (!imu_reported_) {
      imu_reported_ = true;
      RCLCPP_INFO(get_logger(), "lidar IMU %.0f Hz", count / 5.0);
    }
    const auto least = clock_.take_least();
    if (least.count("cloud") && least.count("imu")) {
      const double gap = (least.at("cloud") - least.at("imu")) * 1e-6;
      if (std::abs(gap - gap_) > 5.0) {
        gap_ = gap;
        RCLCPP_INFO(get_logger(),
                    "clock: arrival - stamp, least over 5 s: cloud %.1f ms, IMU %.1f ms, gap %.1f ms (same robot clock if small)",
                    least.at("cloud") * 1e-6, least.at("imu") * 1e-6, gap);
      }
    }
  }

  // Per-point 'time' field range: Point-LIO reads it as seconds after the header stamp.
  static std::string time_range(const DdsCloud& m) {
    const auto f = field(m, "time");
    if (!f || m.is_bigendian() || (f->first->datatype() != kFloat32 && f->first->datatype() != kFloat64)) return "";
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    const size_t n = size_t(m.width()) * m.height();
    for (size_t i = 0; i < n; ++i) {
      const uint8_t* p = m.data().data() + i * m.point_step() + f->second;
      double t;
      if (f->first->datatype() == kFloat32) {
        float v;
        std::memcpy(&v, p, 4);
        t = v;
      } else {
        std::memcpy(&t, p, 8);
      }
      lo = std::min(lo, t);
      hi = std::max(hi, t);
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), ", point time %.4f .. %.4f", lo, hi);
    return buf;
  }

  void publish(const DdsCloud& m) {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = clock_.stamp(m.header(), "cloud");
    msg.header.frame_id = frame_;
    msg.height = m.height();
    msg.width = m.width();
    for (const auto& f : m.fields()) {
      sensor_msgs::msg::PointField pf;
      pf.name = f.name();
      pf.offset = f.offset();
      pf.datatype = f.datatype();
      pf.count = f.count();
      msg.fields.push_back(pf);
    }
    msg.is_bigendian = m.is_bigendian();
    msg.point_step = m.point_step();
    msg.row_step = m.row_step();
    msg.data = m.data();
    bool dense = true;
    if (range_offset_ != 0.0 || body_box_) dense = correct(m, msg.data);
    msg.is_dense = m.is_dense() && dense;
    cloud_pub_->publish(msg);
  }

  // In place: every range pulled back by range_offset along its ray (the L1 reports them that much too long),
  // and points inside body_box (base frame: the robot itself) blanked. Points left nearer than kMinRange, and
  // the robot's, become NaN. Returns false if any did.
  bool correct(const DdsCloud& m, std::vector<uint8_t>& data) {
    const auto x = field(m, "x"), y = field(m, "y"), z = field(m, "z");
    if (!x || !y || !z || m.is_bigendian() || x->first->datatype() != kFloat32 || y->first->datatype() != kFloat32 ||
        z->first->datatype() != kFloat32) {
      if (!warned_format_) {
        warned_format_ = true;
        RCLCPP_WARN(get_logger(), "cloud has no little-endian float32 x, y, z: published uncorrected");
      }
      return true;
    }
    const uint32_t off[3] = {x->second, y->second, z->second};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const size_t n = size_t(m.width()) * m.height();
    bool dense = true;
    for (size_t i = 0; i < n; ++i) {
      uint8_t* p = data.data() + i * m.point_step();
      float v[3];
      for (int c = 0; c < 3; ++c) std::memcpy(&v[c], p + off[c], 4);
      if (range_offset_ != 0.0) {
        const double r = std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]);
        const double k = r - range_offset_ >= kMinRange ? 1.0 - range_offset_ / r : std::nan("");
        for (int c = 0; c < 3; ++c) v[c] = static_cast<float>(v[c] * k);
        if (std::isnan(k)) dense = false;
      }
      if (body_box_) {
        const Eigen::Vector3d b = mount_r_ * Eigen::Vector3d(v[0], v[1], v[2]) + mount_t_;
        bool inside = true;
        for (int c = 0; c < 3; ++c) inside = inside && b[c] >= (*body_box_)(c, 0) && b[c] <= (*body_box_)(c, 1);
        if (inside) {
          v[0] = v[1] = v[2] = nan;
          dense = false;
        }
      }
      for (int c = 0; c < 3; ++c) std::memcpy(p + off[c], &v[c], 4);
    }
    return dense;
  }

  RobotClock clock_;
  std::string frame_, base_frame_;
  double range_offset_ = 0.0;
  std::optional<Eigen::Matrix<double, 3, 2>> body_box_;
  Eigen::Matrix3d mount_r_;
  Eigen::Vector3d mount_t_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::TimerBase::SharedPtr watch_timer_, report_timer_;

  unitree::robot::ChannelPublisherPtr<DdsString> switch_;
  unitree::robot::ChannelSubscriberPtr<LowState> lowstate_;
  unitree::robot::ChannelSubscriberPtr<DdsImu> imu_;
  unitree::robot::ChannelSubscriberPtr<DdsCloud> cloud_;

  std::mutex acc_mutex_, cloud_mutex_;
  std::optional<Eigen::Vector3d> body_acc_;
  std::atomic<long> imu_count_{0};
  double last_cloud_ = -1.0, last_switch_ = 0.0, gap_ = 1e9;
  bool stale_ = false, warned_imu_ = false, warned_acc_ = false, imu_reported_ = false, warned_format_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const Paths paths = Paths::find();
  const YAML::Node config = load_config(paths, "real_sensors");
  use_ros_domain(config);
  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, arg(args, "--interface"));
  } catch (const std::exception& e) {
    std::cerr << "[SDK2] Failed to initialize ChannelFactory: " << e.what() << std::endl;
    return 1;
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RealSensors>(args, config);
  rclcpp::spin(node);
  node->close();
  node.reset();
  if (rclcpp::ok()) rclcpp::shutdown();
  // The SDK's DDS tears down in static destructors that can throw; nothing is left to flush but stdio.
  std::fflush(nullptr);
  std::_Exit(0);
}
