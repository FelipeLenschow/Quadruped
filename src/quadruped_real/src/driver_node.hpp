#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include "control_loop.hpp"

namespace quadruped_real {

struct Paths {
  std::string config, policies, geometry, repo;
  static Paths find();
  std::string resolve(const std::string& path) const;  // policies.json paths are relative to the repo
};

// command_safety_processor.HeartbeatStats: gaps between heartbeats over the last 10 s.
class HeartbeatStats {
 public:
  void add(double now);
  std::string summary(double now);  // JSON, "null" before the first heartbeat

 private:
  void trim(double now);
  std::optional<double> first_, last_;
  std::deque<std::tuple<double, double, long>> gaps_;  // arrival, gap, missed
  long missed_total_ = 0, count_ = 0;
};

// Everything ROS: subscriptions become commands for the control loop, its snapshots become topics. Also the
// policy loader, the operator's ENTER, the banners and the head light. Runs on the cores the loop is not on.
class DriverNode : public rclcpp::Node {
 public:
  DriverNode(Paths paths, YAML::Node config, ControlLoop& loop, CommandBox& commands, SnapshotBox& snapshots);
  ~DriverNode() override;

  // The controller with main (the launcher's checkpoint) and the safety policy loaded.
  std::unique_ptr<qc::Controller> make_controller(const std::string& checkpoint);

 private:
  struct CatalogEntry {
    std::string path;
    qc::PolicyEntry entry;
  };

  void send(Command c);
  Command at(Command::Kind kind, double now = 0.0);
  std::shared_ptr<qc::PolicyRunner> load_runner(const std::string& name, const std::string& checkpoint,
                                                qc::PolicyEntry& entry);
  void load_in_background(const std::string& name);
  void drain_snapshots();
  void handle(const Snapshot& s);
  void publish_telemetry(const Snapshot& s);
  void publish_robot_state(const Snapshot& s);
  void publish_calibration();
  void stdin_loop();
  void update_brightness();
  void light_worker();

  Paths paths_;
  YAML::Node config_;
  ControlLoop& loop_;
  CommandBox& commands_;
  SnapshotBox& snapshots_;
  std::vector<std::pair<std::string, CatalogEntry>> catalog_;
  std::string safety_policy_;
  qc::Kinematics kinematics_;
  std::array<std::string, 12> joint_names_;

  std::mutex loading_mutex_;
  std::set<std::string> loading_;

  HeartbeatStats heartbeat_stats_;
  std::mutex stats_mutex_;
  std::atomic<bool> estop_{false}, blocked_{false}, stdin_ok_{true};
  bool last_estop_ = false;
  double state_period_ = 0.1, state_time_ = 0.0;
  double odom_height_ = 0.33;
  std::optional<Eigen::Vector3d> odom_pos_;
  std::optional<double> odom_time_;
  std::optional<double> first_state_time_;
  bool console_checked_ = false;
  Eigen::Vector4d fsr_offset_ = Eigen::Vector4d::Zero();
  double contact_threshold_ = 0.0;
  bool have_calibration_ = false;
  int brightness_ = -1;
  std::mutex light_mutex_;
  std::condition_variable light_cv_;
  std::optional<int> light_request_;
  bool light_stop_ = false;
  std::thread light_thread_;

  std::vector<rclcpp::SubscriptionBase::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_, cmd_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_sim_pub_, odom_est_pub_, odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr gravity_pub_, lin_vel_pub_, ang_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr height_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr contact_pub_, leg_odom_pub_, foot_force_pub_,
      foot_cal_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_, pose_status_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_state_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;
  rclcpp::TimerBase::SharedPtr drain_timer_, calibration_timer_, light_timer_;
  std::thread stdin_thread_;
};

}  // namespace quadruped_real
