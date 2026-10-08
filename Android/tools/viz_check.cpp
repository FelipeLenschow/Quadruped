// Subscribes to the viz topics the app uses and prints what decoded; publishes one goal.
#include "ros_link.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

int main(int argc, char** argv) {
    roslink::LinkConfig cfg;
    cfg.domain_id = std::atoi(argv[1]);
    if (argc > 2) cfg.local_ip = argv[2];
    roslink::Link link;
    if (!link.start(cfg)) return 1;
    using T = roslink::MsgType;
    link.subscribe("/map", T::OccupancyGrid, true, true);
    link.subscribe("/tf_static", T::TFMessage, true, true);
    link.subscribe("/tf", T::TFMessage, false);
    link.subscribe("/scan", T::LaserScan, false);
    link.subscribe("/plan", T::Path);
    link.subscribe("/sensors/joint_states", T::JointState, false);
    link.advertise("/goal_pose", T::PoseStamped);
    link.subscribe("/lio/map_voxels", T::PointCloud2, true, true);
    link.subscribe("/lio/map_voxels/delta", T::PointCloud2, true, false, true);
    link.subscribe("/lio/odom", T::Odometry, false);
    size_t delta_points = 0, delta_takes = 0;
    float last_delta[3] = {};
    auto drain = [&] {
        auto b = link.take_queued("/lio/map_voxels/delta");
        if (b.empty()) return;
        delta_points += b.size() / 12;
        delta_takes++;
        std::memcpy(last_delta, b.data() + b.size() - 12, 12);
    };
    for (int i = 0; i < 8; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        drain();
    }
    link.publish_pose2d("/goal_pose", "map", 1.5, -2.0, 1.5707963);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto m = link.latest("/map");
    std::printf("\nmap #%llu frame=%s meta=", (unsigned long long)m.count, m.text.c_str());
    for (double v : m.values) std::printf("%g ", v);
    std::printf("cells=%zu first=%d,%d,%d\n", m.bytes ? m.bytes->size() : 0, m.bytes ? (*m.bytes)[0] : 9,
                m.bytes ? (*m.bytes)[1] : 9, m.bytes ? (*m.bytes)[2] : 9);
    auto s = link.latest("/scan");
    std::printf("scan #%llu frame=%s n=%zu head=", (unsigned long long)s.count, s.text.c_str(), s.values.size());
    for (size_t i = 0; i < s.values.size() && i < 5; ++i) std::printf("%g ", s.values[i]);
    auto p = link.latest("/plan");
    std::printf("plan #%llu frame=%s xy=", (unsigned long long)p.count, p.text.c_str());
    for (double v : p.values) std::printf("%g ", v);
    auto j = link.latest("/sensors/joint_states");
    std::printf("\njoints #%llu names=%s q=", (unsigned long long)j.count, j.text.c_str());
    for (double v : j.values) std::printf("%g ", v);
    std::printf("\n");
    std::printf("tf:\n%s", link.tf_snapshot().c_str());
    auto c = link.latest("/lio/map_voxels");
    const float* f = c.bytes && c.bytes->size() >= 12 ? reinterpret_cast<const float*>(c.bytes->data()) : nullptr;
    std::printf("lio snapshot #%llu frame=%s points=%g bytes=%zu first=(%g %g %g)\n", (unsigned long long)c.count,
                c.text.c_str(), c.values.empty() ? 0.0 : c.values[0], c.bytes ? c.bytes->size() : 0,
                f ? f[0] : 0, f ? f[1] : 0, f ? f[2] : 0);
    drain();
    std::printf("lio deltas: %zu points in %zu takes, #%llu samples, last=(%g %g %g)\n", delta_points, delta_takes,
                (unsigned long long)link.latest("/lio/map_voxels/delta").count, last_delta[0], last_delta[1], last_delta[2]);
    auto o = link.latest("/lio/odom");
    std::printf("lio odom #%llu frames=%s pose=", (unsigned long long)o.count, o.text.c_str());
    for (double v : o.values) std::printf("%.3f ", v);
    std::printf("\n");
    link.stop();
}
