// A single DDS participant that speaks ROS 2 (Humble, rmw_fastrtps) without rclcpp.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace roslink {

// Kotlin's MsgType mirrors this order.
enum class MsgType {
    Float32, Bool, String, Vector3, Float32MultiArray,
    JointState, TFMessage, OccupancyGrid, Path, LaserScan,
    Twist, PoseStamped, PointCloud2, Odometry,
};

struct LinkConfig {
    int domain_id = 42;
    // "ip:port" of a Fast DDS discovery server. Empty = plain multicast discovery.
    std::string discovery_server;
    // Only this address (plus loopback) is used and advertised. Empty = every interface.
    std::string local_ip;
};

// Latest sample of a topic, flattened:
//   Float32/Bool/Vector3/Float32MultiArray: values
//   String: text
//   JointState: text = names joined by ',', values = positions
//   OccupancyGrid: text = frame, values = [width, height, resolution, ox, oy, oz, qx, qy, qz, qw], bytes = cells
//   Path: text = frame, values = [x0, y0, x1, y1, ...]
//   LaserScan: text = frame, values = [angle_min, angle_increment, range_min, range_max, ranges...]
//   PointCloud2: text = frame, values = [points], bytes = x,y,z float32 little-endian per point
//   Odometry: text = "frame child", values = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
//   TFMessage: nothing here, see tf_snapshot()
struct Latest {
    std::string text;
    std::vector<double> values;
    std::shared_ptr<const std::vector<int8_t>> bytes;
    int64_t recv_ns = 0;  // steady clock; 0 = nothing yet
    uint64_t count = 0;
};

class Link {
public:
    Link();
    ~Link();

    bool start(const LinkConfig& cfg, std::string* error = nullptr);
    void stop();
    bool running() const;

    // Creates the writer now, so discovery is done before the first publish
    // (a new writer's first samples go nowhere until it has matched).
    bool advertise(const std::string& topic, MsgType type);
    bool publish_float32(const std::string& topic, float v);
    bool publish_bool(const std::string& topic, bool v);
    bool publish_string(const std::string& topic, const std::string& v);
    bool publish_twist(const std::string& topic, double vx, double vy, double wz);
    bool publish_pose2d(const std::string& topic, const std::string& frame, double x, double y, double yaw);

    // reliable=false for streams (no retransmits competing with the heartbeat);
    // transient_local for latched topics such as /map and /tf_static;
    // queue: also keep every sample's bytes until take_queued(), for topics where each one counts.
    bool subscribe(const std::string& topic, MsgType type, bool reliable = true, bool transient_local = false,
                   bool queue = false);
    void unsubscribe(const std::string& topic);
    Latest latest(const std::string& topic) const;
    // The bytes of every sample since the last call, concatenated, and how many were dropped
    // because the queue was full.
    std::vector<int8_t> take_queued(const std::string& topic, uint64_t* dropped = nullptr);
    // One line per known frame: "child parent tx ty tz qx qy qz qw age_ms"
    std::string tf_snapshot() const;
    // Readers matched to our writer on this topic, -1 if we have no writer on it.
    int matched_readers(const std::string& topic) const;

    static int64_t now_ns();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace roslink
