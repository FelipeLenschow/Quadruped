// Runs the app's DDS core on a PC against ROS 2 Humble, to check types, QoS and
// discovery without a phone:  ./link_check.sh domain [server_ip:port|""] [local_ip]
#include "ros_link.hpp"

#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    roslink::LinkConfig cfg;
    if (argc < 2) {
        std::printf("usage: link_check domain [server_ip:port|\"\"] [local_ip]\n");
        return 2;
    }
    cfg.domain_id = std::atoi(argv[1]);
    if (argc > 2) cfg.discovery_server = argv[2];
    if (argc > 3) cfg.local_ip = argv[3];
    roslink::Link link;
    std::string err;
    if (!link.start(cfg, &err)) {
        std::printf("start failed: %s\n", err.c_str());
        return 1;
    }
    link.subscribe("/robot_state", roslink::MsgType::String);
    link.subscribe("/safety/estop_state", roslink::MsgType::Bool);
    link.subscribe("/estimator/base_lin_vel", roslink::MsgType::Vector3);
    link.subscribe("/estimator/feet_contact", roslink::MsgType::Float32MultiArray);
    for (int i = 0; i < 100; ++i) {
        link.publish_float32("/safety/heartbeat", static_cast<float>(i));
        link.publish_string("/pipeline/mode", "pose");
        link.publish_twist("/cmd_vel/phone", 0.5, -0.4, 0.25);
        if (i % 10 == 0) {
            auto rs = link.latest("/robot_state");
            auto es = link.latest("/safety/estop_state");
            auto v = link.latest("/estimator/base_lin_vel");
            auto c = link.latest("/estimator/feet_contact");
            std::printf("t=%ds heartbeat_readers=%d robot_state#%llu '%s' estop#%llu=%g vel#%llu=%zu contact#%llu=%zu\n",
                        i / 10, link.matched_readers("/safety/heartbeat"),
                        (unsigned long long)rs.count, rs.text.c_str(),
                        (unsigned long long)es.count, es.values.empty() ? -1.0 : es.values[0],
                        (unsigned long long)v.count, v.values.size(),
                        (unsigned long long)c.count, c.values.size());
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    link.stop();
}
