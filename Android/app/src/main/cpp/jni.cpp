#include "ros_link.hpp"

#include <jni.h>

#include <string>

namespace {

roslink::Link g_link;

std::string str(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace

#define FN(ret, name) extern "C" JNIEXPORT ret JNICALL Java_com_quadruped_console_NativeLink_##name

FN(jstring, nativeStart)(JNIEnv* env, jobject, jint domain, jstring server, jstring localIp) {
    roslink::LinkConfig cfg;
    cfg.domain_id = domain;
    cfg.discovery_server = str(env, server);
    cfg.local_ip = str(env, localIp);
    std::string err;
    if (g_link.start(cfg, &err)) return nullptr;
    return env->NewStringUTF(err.c_str());
}

FN(void, nativeStop)(JNIEnv*, jobject) { g_link.stop(); }

FN(jboolean, publishFloat32)(JNIEnv* env, jobject, jstring topic, jfloat v) {
    return g_link.publish_float32(str(env, topic), v);
}

FN(jboolean, publishBool)(JNIEnv* env, jobject, jstring topic, jboolean v) {
    return g_link.publish_bool(str(env, topic), v);
}

FN(jboolean, publishString)(JNIEnv* env, jobject, jstring topic, jstring v) {
    return g_link.publish_string(str(env, topic), str(env, v));
}

FN(jboolean, publishTwist)(JNIEnv* env, jobject, jstring topic, jdouble vx, jdouble vy, jdouble wz) {
    return g_link.publish_twist(str(env, topic), vx, vy, wz);
}

FN(jboolean, publishPose2d)(JNIEnv* env, jobject, jstring topic, jstring frame, jdouble x, jdouble y, jdouble yaw) {
    return g_link.publish_pose2d(str(env, topic), str(env, frame), x, y, yaw);
}

FN(jboolean, nativeAdvertise)(JNIEnv* env, jobject, jstring topic, jint type) {
    return g_link.advertise(str(env, topic), static_cast<roslink::MsgType>(type));
}

FN(jboolean, nativeSubscribe)(JNIEnv* env, jobject, jstring topic, jint type, jboolean reliable, jboolean transientLocal) {
    return g_link.subscribe(str(env, topic), static_cast<roslink::MsgType>(type), reliable, transientLocal);
}

FN(void, unsubscribe)(JNIEnv* env, jobject, jstring topic) { g_link.unsubscribe(str(env, topic)); }

FN(jbyteArray, latestBytes)(JNIEnv* env, jobject, jstring topic) {
    auto l = g_link.latest(str(env, topic));
    if (!l.bytes) return nullptr;
    jbyteArray a = env->NewByteArray(static_cast<jsize>(l.bytes->size()));
    env->SetByteArrayRegion(a, 0, static_cast<jsize>(l.bytes->size()), reinterpret_cast<const jbyte*>(l.bytes->data()));
    return a;
}

FN(jstring, tfSnapshot)(JNIEnv* env, jobject) { return env->NewStringUTF(g_link.tf_snapshot().c_str()); }

FN(jstring, latestText)(JNIEnv* env, jobject, jstring topic) {
    auto l = g_link.latest(str(env, topic));
    return l.count ? env->NewStringUTF(l.text.c_str()) : nullptr;
}

FN(jdoubleArray, latestValues)(JNIEnv* env, jobject, jstring topic) {
    auto l = g_link.latest(str(env, topic));
    if (!l.count) return nullptr;
    jdoubleArray a = env->NewDoubleArray(static_cast<jsize>(l.values.size()));
    env->SetDoubleArrayRegion(a, 0, static_cast<jsize>(l.values.size()), l.values.data());
    return a;
}

FN(jlong, latestAgeMs)(JNIEnv* env, jobject, jstring topic) {
    auto l = g_link.latest(str(env, topic));
    return l.count ? (roslink::Link::now_ns() - l.recv_ns) / 1000000 : -1;
}

FN(jlong, latestCount)(JNIEnv* env, jobject, jstring topic) {
    return static_cast<jlong>(g_link.latest(str(env, topic)).count);
}

FN(jint, matchedReaders)(JNIEnv* env, jobject, jstring topic) {
    return g_link.matched_readers(str(env, topic));
}
