// Hand-written CDR type support for the ROS messages the console uses.
// Names and layouts must match what rmw_fastrtps puts on the wire for ROS 2 Humble.
#pragma once

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>
#include <fastdds/dds/topic/TopicDataType.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace roslink {

struct Float32 {
    static constexpr const char* kName = "std_msgs::msg::dds_::Float32_";
    float data = 0.f;
    void ser(eprosima::fastcdr::Cdr& c) const { c << data; }
    void deser(eprosima::fastcdr::Cdr& c) { c >> data; }
    size_t bound() const { return 4; }
};

struct Bool {
    static constexpr const char* kName = "std_msgs::msg::dds_::Bool_";
    bool data = false;
    void ser(eprosima::fastcdr::Cdr& c) const { c << data; }
    void deser(eprosima::fastcdr::Cdr& c) { c >> data; }
    size_t bound() const { return 1; }
};

struct String {
    static constexpr const char* kName = "std_msgs::msg::dds_::String_";
    std::string data;
    void ser(eprosima::fastcdr::Cdr& c) const { c << data; }
    void deser(eprosima::fastcdr::Cdr& c) { c >> data; }
    size_t bound() const { return 4 + data.size() + 1; }
};

struct Vector3 {
    static constexpr const char* kName = "geometry_msgs::msg::dds_::Vector3_";
    double x = 0, y = 0, z = 0;
    void ser(eprosima::fastcdr::Cdr& c) const { c << x << y << z; }
    void deser(eprosima::fastcdr::Cdr& c) { c >> x >> y >> z; }
    size_t bound() const { return 8 + 24; }
};

struct Twist {
    static constexpr const char* kName = "geometry_msgs::msg::dds_::Twist_";
    Vector3 linear, angular;
    void ser(eprosima::fastcdr::Cdr& c) const { linear.ser(c); angular.ser(c); }
    void deser(eprosima::fastcdr::Cdr& c) { linear.deser(c); angular.deser(c); }
    size_t bound() const { return 8 + 48; }
};

struct Float32MultiArray {
    static constexpr const char* kName = "std_msgs::msg::dds_::Float32MultiArray_";
    struct Dim {
        std::string label;
        uint32_t size = 0, stride = 0;
    };
    std::vector<Dim> dim;
    uint32_t data_offset = 0;
    std::vector<float> data;

    void ser(eprosima::fastcdr::Cdr& c) const {
        c << static_cast<uint32_t>(dim.size());
        for (const auto& d : dim) c << d.label << d.size << d.stride;
        c << data_offset << data;
    }
    void deser(eprosima::fastcdr::Cdr& c) {
        uint32_t n = 0;
        c >> n;
        dim.resize(n);
        for (auto& d : dim) c >> d.label >> d.size >> d.stride;
        c >> data_offset >> data;
    }
    size_t bound() const {
        size_t s = 8;
        for (const auto& d : dim) s += 4 + d.label.size() + 1 + 3 + 8;
        return s + 8 + 4 * data.size() + 8;
    }
};

struct Header {
    int32_t sec = 0;
    uint32_t nanosec = 0;
    std::string frame_id;
    void ser(eprosima::fastcdr::Cdr& c) const { c << sec << nanosec << frame_id; }
    void deser(eprosima::fastcdr::Cdr& c) { c >> sec >> nanosec >> frame_id; }
};

struct Pose {
    double px = 0, py = 0, pz = 0, qx = 0, qy = 0, qz = 0, qw = 1;
    void ser(eprosima::fastcdr::Cdr& c) const { c << px << py << pz << qx << qy << qz << qw; }
    void deser(eprosima::fastcdr::Cdr& c) { c >> px >> py >> pz >> qx >> qy >> qz >> qw; }
};

struct PoseStamped {
    static constexpr const char* kName = "geometry_msgs::msg::dds_::PoseStamped_";
    Header header;
    Pose pose;
    void ser(eprosima::fastcdr::Cdr& c) const { header.ser(c); pose.ser(c); }
    void deser(eprosima::fastcdr::Cdr& c) { header.deser(c); pose.deser(c); }
    size_t bound() const { return 8 + 4 + header.frame_id.size() + 1 + 8 + 56; }
};

struct JointState {
    static constexpr const char* kName = "sensor_msgs::msg::dds_::JointState_";
    Header header;
    std::vector<std::string> name;
    std::vector<double> position, velocity, effort;
    void deser(eprosima::fastcdr::Cdr& c) {
        header.deser(c);
        uint32_t n = 0;
        c >> n;
        name.resize(n);
        for (auto& s : name) c >> s;
        c >> position >> velocity >> effort;
    }
    void ser(eprosima::fastcdr::Cdr&) const {}
    size_t bound() const { return 0; }
};

struct TransformStamped {
    Header header;
    std::string child_frame_id;
    double tx = 0, ty = 0, tz = 0, qx = 0, qy = 0, qz = 0, qw = 1;
    void deser(eprosima::fastcdr::Cdr& c) {
        header.deser(c);
        c >> child_frame_id >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    }
};

struct TFMessage {
    static constexpr const char* kName = "tf2_msgs::msg::dds_::TFMessage_";
    std::vector<TransformStamped> transforms;
    void deser(eprosima::fastcdr::Cdr& c) {
        uint32_t n = 0;
        c >> n;
        transforms.resize(n);
        for (auto& t : transforms) t.deser(c);
    }
    void ser(eprosima::fastcdr::Cdr&) const {}
    size_t bound() const { return 0; }
};

struct OccupancyGrid {
    static constexpr const char* kName = "nav_msgs::msg::dds_::OccupancyGrid_";
    Header header;
    int32_t load_sec = 0;
    uint32_t load_nanosec = 0;
    float resolution = 0;
    uint32_t width = 0, height = 0;
    Pose origin;
    std::vector<int8_t> data;
    void deser(eprosima::fastcdr::Cdr& c) {
        header.deser(c);
        c >> load_sec >> load_nanosec >> resolution >> width >> height;
        origin.deser(c);
        c >> data;
    }
    void ser(eprosima::fastcdr::Cdr&) const {}
    size_t bound() const { return 0; }
};

struct Path {
    static constexpr const char* kName = "nav_msgs::msg::dds_::Path_";
    Header header;
    std::vector<PoseStamped> poses;
    void deser(eprosima::fastcdr::Cdr& c) {
        header.deser(c);
        uint32_t n = 0;
        c >> n;
        poses.resize(n);
        for (auto& p : poses) p.deser(c);
    }
    void ser(eprosima::fastcdr::Cdr&) const {}
    size_t bound() const { return 0; }
};

struct LaserScan {
    static constexpr const char* kName = "sensor_msgs::msg::dds_::LaserScan_";
    Header header;
    float angle_min = 0, angle_max = 0, angle_increment = 0, time_increment = 0, scan_time = 0;
    float range_min = 0, range_max = 0;
    std::vector<float> ranges, intensities;
    void deser(eprosima::fastcdr::Cdr& c) {
        header.deser(c);
        c >> angle_min >> angle_max >> angle_increment >> time_increment >> scan_time >> range_min >> range_max;
        c >> ranges >> intensities;
    }
    void ser(eprosima::fastcdr::Cdr&) const {}
    size_t bound() const { return 0; }
};

template <class T>
class RosType : public eprosima::fastdds::dds::TopicDataType {
public:
    RosType() {
        setName(T::kName);
        m_typeSize = 4 + 1024;  // grows on demand (PREALLOCATED_WITH_REALLOC)
        m_isGetKeyDefined = false;
        auto_fill_type_object(false);
        auto_fill_type_information(false);
    }

    bool serialize(void* data, eprosima::fastrtps::rtps::SerializedPayload_t* payload) override {
        eprosima::fastcdr::FastBuffer buf(reinterpret_cast<char*>(payload->data), payload->max_size);
        eprosima::fastcdr::Cdr c(buf, eprosima::fastcdr::Cdr::DEFAULT_ENDIAN, eprosima::fastcdr::Cdr::DDS_CDR);
        payload->encapsulation = c.endianness() == eprosima::fastcdr::Cdr::BIG_ENDIANNESS ? CDR_BE : CDR_LE;
        try {
            c.serialize_encapsulation();
            static_cast<T*>(data)->ser(c);
        } catch (...) {
            return false;
        }
        payload->length = static_cast<uint32_t>(c.getSerializedDataLength());
        return true;
    }

    bool deserialize(eprosima::fastrtps::rtps::SerializedPayload_t* payload, void* data) override {
        eprosima::fastcdr::FastBuffer buf(reinterpret_cast<char*>(payload->data), payload->length);
        eprosima::fastcdr::Cdr c(buf, eprosima::fastcdr::Cdr::DEFAULT_ENDIAN, eprosima::fastcdr::Cdr::DDS_CDR);
        try {
            c.read_encapsulation();
            payload->encapsulation = c.endianness() == eprosima::fastcdr::Cdr::BIG_ENDIANNESS ? CDR_BE : CDR_LE;
            static_cast<T*>(data)->deser(c);
        } catch (...) {
            return false;
        }
        return true;
    }

    std::function<uint32_t()> getSerializedSizeProvider(void* data) override {
        return [data]() { return static_cast<uint32_t>(4 + static_cast<T*>(data)->bound() + 8); };
    }

    void* createData() override { return new T(); }
    void deleteData(void* data) override { delete static_cast<T*>(data); }

    bool getKey(void*, eprosima::fastrtps::rtps::InstanceHandle_t*, bool) override { return false; }
};

}  // namespace roslink
