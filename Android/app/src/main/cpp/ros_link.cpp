#include "ros_link.hpp"
#include "ros_types.hpp"

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/rtps/attributes/ServerAttributes.h>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.h>
#include <fastrtps/utils/IPLocator.h>

#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <sstream>

using namespace eprosima::fastdds::dds;
using eprosima::fastrtps::rtps::IPLocator;
using eprosima::fastrtps::rtps::Locator_t;

namespace roslink {

namespace {

struct Tf {
    std::string parent;
    double t[3], q[4];
    int64_t recv_ns;
};

struct Store {
    std::mutex mtx;
    std::map<std::string, Latest> latest;
    std::map<std::string, Tf> tf;
};

void to_latest(const Float32& m, Latest& l, Store&) { l.values = {m.data}; }
void to_latest(const Bool& m, Latest& l, Store&) { l.values = {m.data ? 1.0 : 0.0}; }
void to_latest(const String& m, Latest& l, Store&) { l.text = m.data; }
void to_latest(const Vector3& m, Latest& l, Store&) { l.values = {m.x, m.y, m.z}; }
void to_latest(const Float32MultiArray& m, Latest& l, Store&) { l.values.assign(m.data.begin(), m.data.end()); }

void to_latest(const JointState& m, Latest& l, Store&) {
    l.text.clear();
    for (size_t i = 0; i < m.name.size(); ++i) l.text += (i ? "," : "") + m.name[i];
    l.values = m.position;
}

void to_latest(const TFMessage& m, Latest&, Store& s) {
    int64_t now = Link::now_ns();
    for (const auto& t : m.transforms) {
        s.tf[t.child_frame_id] = Tf{t.header.frame_id, {t.tx, t.ty, t.tz}, {t.qx, t.qy, t.qz, t.qw}, now};
    }
}

void to_latest(const OccupancyGrid& m, Latest& l, Store&) {
    l.text = m.header.frame_id;
    const Pose& o = m.origin;
    l.values = {double(m.width), double(m.height), m.resolution, o.px, o.py, o.pz, o.qx, o.qy, o.qz, o.qw};
    l.bytes = std::make_shared<const std::vector<int8_t>>(m.data);
}

void to_latest(const Path& m, Latest& l, Store&) {
    l.text = m.header.frame_id;
    l.values.clear();
    l.values.reserve(m.poses.size() * 2);
    for (const auto& p : m.poses) {
        l.values.push_back(p.pose.px);
        l.values.push_back(p.pose.py);
    }
}

void to_latest(const LaserScan& m, Latest& l, Store&) {
    l.text = m.header.frame_id;
    l.values = {m.angle_min, m.angle_increment, m.range_min, m.range_max};
    l.values.insert(l.values.end(), m.ranges.begin(), m.ranges.end());
}

// The ROS 2 defaults (rmw_qos_profile_default): reliable, volatile, keep last 10.
template <class Q>
void ros_qos(Q& q, bool reliable = true, bool transient_local = false) {
    q.reliability().kind = reliable ? RELIABLE_RELIABILITY_QOS : BEST_EFFORT_RELIABILITY_QOS;
    q.durability().kind = transient_local ? TRANSIENT_LOCAL_DURABILITY_QOS : VOLATILE_DURABILITY_QOS;
    q.history().kind = KEEP_LAST_HISTORY_QOS;
    q.history().depth = transient_local ? 1 : 10;
    q.endpoint().history_memory_policy = eprosima::fastrtps::rtps::PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
    q.data_sharing().off();
}

bool parse_ip_port(const std::string& s, std::string& ip, uint32_t& port) {
    auto colon = s.rfind(':');
    ip = s.substr(0, colon);
    port = 11811;
    if (colon != std::string::npos) {
        try {
            port = static_cast<uint32_t>(std::stoul(s.substr(colon + 1)));
        } catch (...) {
            return false;
        }
    }
    return IPLocator::isIPv4(ip);
}

}  // namespace

struct Link::Impl {
    template <class T>
    struct Listener : DataReaderListener {
        Store* store;
        std::string topic;
        void on_data_available(DataReader* reader) override {
            T msg;
            SampleInfo info;
            while (reader->take_next_sample(&msg, &info) == ReturnCode_t::RETCODE_OK) {
                if (!info.valid_data) continue;
                std::lock_guard<std::mutex> lk(store->mtx);
                Latest& l = store->latest[topic];
                to_latest(msg, l, *store);
                l.recv_ns = Link::now_ns();
                l.count++;
            }
        }
    };

    std::mutex mtx;
    DomainParticipant* participant = nullptr;
    Publisher* publisher = nullptr;
    Subscriber* subscriber = nullptr;
    std::map<std::string, Topic*> topics;
    std::map<std::string, DataWriter*> writers;
    std::map<std::string, std::pair<DataReader*, std::unique_ptr<DataReaderListener>>> readers;
    Store store;

    Topic* topic(const std::string& name, const char* type) {
        auto it = topics.find(name);
        if (it != topics.end()) return it->second;
        Topic* t = participant->create_topic("rt" + name, type, TOPIC_QOS_DEFAULT);
        if (t) topics[name] = t;
        return t;
    }

    template <class T>
    DataWriter* writer(const std::string& name) {
        auto it = writers.find(name);
        if (it != writers.end()) return it->second;
        Topic* t = topic(name, T::kName);
        if (!t) return nullptr;
        DataWriterQos q = DATAWRITER_QOS_DEFAULT;
        ros_qos(q);
        q.publish_mode().kind = SYNCHRONOUS_PUBLISH_MODE;
        DataWriter* w = publisher->create_datawriter(t, q);
        if (w) writers[name] = w;
        return w;
    }

    template <class T>
    bool publish(const std::string& name, T& msg) {
        // Held across write() so stop() cannot delete the writer under us; with
        // KEEP_LAST the write never blocks.
        std::lock_guard<std::mutex> lk(mtx);
        if (!participant) return false;
        DataWriter* w = writer<T>(name);
        return w && w->write(&msg);
    }

    template <class T>
    bool subscribe(const std::string& name, bool reliable, bool transient_local) {
        if (readers.count(name)) return true;
        Topic* t = topic(name, T::kName);
        if (!t) return false;
        auto l = std::make_unique<Listener<T>>();
        l->store = &store;
        l->topic = name;
        DataReaderQos q = DATAREADER_QOS_DEFAULT;
        ros_qos(q, reliable, transient_local);
        DataReader* r = subscriber->create_datareader(t, q, l.get());
        if (!r) return false;
        readers[name] = {r, std::move(l)};
        return true;
    }
};

Link::Link() : impl_(new Impl) {}
Link::~Link() { stop(); }

int64_t Link::now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool Link::start(const LinkConfig& cfg, std::string* error) {
    auto fail = [error](const std::string& e) {
        if (error) *error = e;
        return false;
    };
    stop();
    std::lock_guard<std::mutex> lk(impl_->mtx);

    DomainParticipantQos pq = PARTICIPANT_QOS_DEFAULT;
    pq.name("quadruped_console_android");

    if (!cfg.discovery_server.empty()) {
        std::string ip;
        uint32_t port;
        if (!parse_ip_port(cfg.discovery_server, ip, port)) return fail("bad discovery server: " + cfg.discovery_server);
        // Same role as ROS_SUPER_CLIENT=TRUE in Tools/ros_env.sh: the server sends
        // us every participant, so `ros2 topic list`-style views stay complete.
        auto& disc = pq.wire_protocol().builtin.discovery_config;
        disc.discoveryProtocol = eprosima::fastrtps::rtps::DiscoveryProtocol::SUPER_CLIENT;
        eprosima::fastdds::rtps::RemoteServerAttributes server;
        eprosima::fastdds::rtps::get_server_client_default_guidPrefix(0, server.guidPrefix);
        Locator_t loc;
        IPLocator::setIPv4(loc, ip);
        loc.port = port;
        server.metatrafficUnicastLocatorList.push_back(loc);
        disc.m_DiscoveryServers.push_back(server);
    }

    // UDP only: no shared memory on Android, and pinning the interface stops us
    // advertising the mobile-data address the robot cannot route to.
    auto udp = std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();
    // Loopback stays in, as in Tools/ros_env.sh: Fast DDS sends to participants on
    // the same host over it, so without it a pinned link cannot reach them.
    if (!cfg.local_ip.empty()) {
        udp->interfaceWhiteList.push_back(cfg.local_ip);
        udp->interfaceWhiteList.push_back("127.0.0.1");
    }
    pq.transport().user_transports.push_back(udp);
    pq.transport().use_builtin_transports = false;

    auto* f = DomainParticipantFactory::get_instance();
    impl_->participant = f->create_participant(cfg.domain_id, pq);
    if (!impl_->participant) return fail("create_participant failed");

    for (TypeSupport ts : {TypeSupport(new RosType<Float32>()), TypeSupport(new RosType<Bool>()),
                           TypeSupport(new RosType<String>()), TypeSupport(new RosType<Vector3>()),
                           TypeSupport(new RosType<Twist>()), TypeSupport(new RosType<Float32MultiArray>()),
                           TypeSupport(new RosType<PoseStamped>()), TypeSupport(new RosType<JointState>()),
                           TypeSupport(new RosType<TFMessage>()), TypeSupport(new RosType<OccupancyGrid>()),
                           TypeSupport(new RosType<Path>()), TypeSupport(new RosType<LaserScan>())}) {
        ts.register_type(impl_->participant);
    }
    impl_->publisher = impl_->participant->create_publisher(PUBLISHER_QOS_DEFAULT);
    impl_->subscriber = impl_->participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT);
    if (!impl_->publisher || !impl_->subscriber) return fail("create publisher/subscriber failed");
    return true;
}

void Link::stop() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->participant) return;
    impl_->participant->delete_contained_entities();
    DomainParticipantFactory::get_instance()->delete_participant(impl_->participant);
    impl_->participant = nullptr;
    impl_->publisher = nullptr;
    impl_->subscriber = nullptr;
    impl_->topics.clear();
    impl_->writers.clear();
    impl_->readers.clear();
    std::lock_guard<std::mutex> lk2(impl_->store.mtx);
    impl_->store.latest.clear();
    impl_->store.tf.clear();
}

bool Link::running() const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->participant != nullptr;
}

bool Link::advertise(const std::string& topic, MsgType type) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->participant) return false;
    switch (type) {
        case MsgType::Float32: return impl_->writer<Float32>(topic) != nullptr;
        case MsgType::Bool: return impl_->writer<Bool>(topic) != nullptr;
        case MsgType::String: return impl_->writer<String>(topic) != nullptr;
        case MsgType::Twist: return impl_->writer<Twist>(topic) != nullptr;
        case MsgType::PoseStamped: return impl_->writer<PoseStamped>(topic) != nullptr;
        default: return false;
    }
}

bool Link::publish_float32(const std::string& topic, float v) {
    Float32 m;
    m.data = v;
    return impl_->publish(topic, m);
}

bool Link::publish_bool(const std::string& topic, bool v) {
    Bool m;
    m.data = v;
    return impl_->publish(topic, m);
}

bool Link::publish_string(const std::string& topic, const std::string& v) {
    String m;
    m.data = v;
    return impl_->publish(topic, m);
}

bool Link::publish_twist(const std::string& topic, double vx, double vy, double wz) {
    Twist m;
    m.linear.x = vx;
    m.linear.y = vy;
    m.angular.z = wz;
    return impl_->publish(topic, m);
}

bool Link::publish_pose2d(const std::string& topic, const std::string& frame, double x, double y, double yaw) {
    PoseStamped m;
    // Zero stamp: tf2 reads it as "latest available", which is what a goal from a
    // phone whose clock is not the robot's needs.
    m.header.frame_id = frame;
    m.pose.px = x;
    m.pose.py = y;
    m.pose.qz = std::sin(yaw / 2);
    m.pose.qw = std::cos(yaw / 2);
    return impl_->publish(topic, m);
}

bool Link::subscribe(const std::string& topic, MsgType type, bool reliable, bool transient_local) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->participant) return false;
    switch (type) {
        case MsgType::Float32: return impl_->subscribe<Float32>(topic, reliable, transient_local);
        case MsgType::Bool: return impl_->subscribe<Bool>(topic, reliable, transient_local);
        case MsgType::String: return impl_->subscribe<String>(topic, reliable, transient_local);
        case MsgType::Vector3: return impl_->subscribe<Vector3>(topic, reliable, transient_local);
        case MsgType::Float32MultiArray: return impl_->subscribe<Float32MultiArray>(topic, reliable, transient_local);
        case MsgType::JointState: return impl_->subscribe<JointState>(topic, reliable, transient_local);
        case MsgType::TFMessage: return impl_->subscribe<TFMessage>(topic, reliable, transient_local);
        case MsgType::OccupancyGrid: return impl_->subscribe<OccupancyGrid>(topic, reliable, transient_local);
        case MsgType::Path: return impl_->subscribe<Path>(topic, reliable, transient_local);
        case MsgType::LaserScan: return impl_->subscribe<LaserScan>(topic, reliable, transient_local);
        default: return false;
    }
}

void Link::unsubscribe(const std::string& topic) {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->readers.find(topic);
    if (it == impl_->readers.end()) return;
    impl_->subscriber->delete_datareader(it->second.first);
    impl_->readers.erase(it);
    std::lock_guard<std::mutex> lk2(impl_->store.mtx);
    impl_->store.latest.erase(topic);
}

Latest Link::latest(const std::string& topic) const {
    std::lock_guard<std::mutex> lk(impl_->store.mtx);
    auto it = impl_->store.latest.find(topic);
    return it == impl_->store.latest.end() ? Latest{} : it->second;
}

std::string Link::tf_snapshot() const {
    std::lock_guard<std::mutex> lk(impl_->store.mtx);
    std::ostringstream out;
    out.precision(9);
    int64_t now = now_ns();
    for (const auto& [child, t] : impl_->store.tf) {
        out << child << ' ' << t.parent << ' ' << t.t[0] << ' ' << t.t[1] << ' ' << t.t[2] << ' ' << t.q[0] << ' '
            << t.q[1] << ' ' << t.q[2] << ' ' << t.q[3] << ' ' << (now - t.recv_ns) / 1000000 << '\n';
    }
    return out.str();
}

int Link::matched_readers(const std::string& topic) const {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    auto it = impl_->writers.find(topic);
    if (it == impl_->writers.end()) return -1;
    PublicationMatchedStatus s;
    it->second->get_publication_matched_status(s);
    return s.current_count;
}

}  // namespace roslink
