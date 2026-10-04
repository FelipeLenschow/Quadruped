// Subscribes to the viz topics the app uses and prints what decoded; publishes one goal.
#include "ros_link.hpp"

#include <cstdio>
#include <cstdlib>
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
    std::this_thread::sleep_for(std::chrono::seconds(4));
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
    link.stop();
}
