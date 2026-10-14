// The Go2's front camera to ROS: one JPEG per videohub request, only while someone subscribes.
// camera_bridge reads it from here for its cameras with source: unitree.
#include <cstdio>
#include <iostream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/video/video_client.hpp>

#include "config_loader.hpp"

using namespace quadruped_real;

namespace {

const char* kTopic = "/go2_camera/image/compressed";  // camera_bridge's GO2_CAMERA_TOPIC
const char* kJpeg = "bgr8; jpeg compressed bgr8";

class Go2Camera : public rclcpp::Node {
 public:
  Go2Camera(const std::string& topic, double rate, double timeout) : Node("go2_camera") {
    pub_ = create_publisher<sensor_msgs::msg::CompressedImage>(topic, 2);
    client_ = std::make_unique<unitree::robot::go2::VideoClient>();
    client_->SetTimeout(static_cast<float>(timeout));
    client_->Init();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] { grab(); });
    RCLCPP_INFO(get_logger(), "videohub -> %s, %.0f Hz while subscribed", topic.c_str(), rate);
  }

  void close() { client_.reset(); }

 private:
  void grab() {
    if (pub_->get_subscription_count() == 0) return;
    sensor_msgs::msg::CompressedImage msg;
    const int32_t code = client_->GetImageSample(msg.data);
    if (code != 0 || msg.data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000, "videohub gave no image (error %d)", code);
      return;
    }
    if (!first_) {
      first_ = true;
      RCLCPP_INFO(get_logger(), "first image (%zu kB)", msg.data.size() / 1024);
    }
    msg.header.stamp = now();
    msg.format = kJpeg;
    pub_->publish(msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_;
  std::unique_ptr<unitree::robot::go2::VideoClient> client_;
  rclcpp::TimerBase::SharedPtr timer_;
  bool first_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  const Paths paths = Paths::find();
  const YAML::Node config = load_config(paths, "go2_camera");
  use_ros_domain(config);

  // Rate and timeout of the first camera config.yaml reads from the Go2 (source: unitree).
  double rate = 15.0, timeout = 1.0;
  const YAML::Node cameras = config["cameras"];
  if (cameras && cameras.IsMap()) {
    for (const auto& cam : cameras) {
      if (get<std::string>(cam.second, "source", "unitree") == "unitree") {
        rate = get<double>(cam.second, "rate", rate);
        timeout = get<double>(cam.second, "timeout", timeout);
        break;
      }
    }
  }
  if (!arg(args, "--rate").empty()) rate = std::stod(arg(args, "--rate"));
  const std::string topic = arg(args, "--topic").empty() ? kTopic : arg(args, "--topic");

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, arg(args, "--interface"));
  } catch (const std::exception& e) {
    std::cerr << "[SDK2] Failed to initialize ChannelFactory: " << e.what() << std::endl;
    return 1;
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Go2Camera>(topic, rate, timeout);
  rclcpp::spin(node);
  node->close();
  node.reset();
  if (rclcpp::ok()) rclcpp::shutdown();
  std::fflush(nullptr);
  std::_Exit(0);
}
