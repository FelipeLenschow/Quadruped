package com.quadruped.console

interface RosLink {
    fun publishFloat32(topic: String, v: Float): Boolean
    fun publishBool(topic: String, v: Boolean): Boolean
    fun publishString(topic: String, v: String): Boolean
    fun publishTwist(topic: String, vx: Double, vy: Double, wz: Double): Boolean
    fun latestText(topic: String): String?
    fun latestValues(topic: String): DoubleArray?
    fun latestAgeMs(topic: String): Long
    fun latestCount(topic: String): Long
    fun matchedReaders(topic: String): Int
}

/** Same order as roslink::MsgType. */
enum class MsgType {
    FLOAT32, BOOL, STRING, VECTOR3, FLOAT32_MULTI_ARRAY,
    JOINT_STATE, TF_MESSAGE, OCCUPANCY_GRID, PATH, LASER_SCAN,
    TWIST, POSE_STAMPED, POINT_CLOUD2, ODOMETRY,
}

/** The C++ DDS participant (app/src/main/cpp). One per process. */
object NativeLink : RosLink {
    init {
        System.loadLibrary("roslink")
    }

    /** Returns null on success, otherwise the error. Empty server = multicast discovery. */
    fun start(domain: Int, server: String, localIp: String): String? = nativeStart(domain, server, localIp)

    fun stop() = nativeStop()

    /**
     * reliable=false for streams; transientLocal for latched topics (/map, /tf_static);
     * queue keeps every sample's bytes for [takeQueued], where none may be missed.
     */
    fun subscribe(
        topic: String, type: MsgType, reliable: Boolean = true, transientLocal: Boolean = false, queue: Boolean = false,
    ) = nativeSubscribe(topic, type.ordinal, reliable, transientLocal, queue)

    /** Create the writer now so it has matched before the first publish. */
    fun advertise(topic: String, type: MsgType) = nativeAdvertise(topic, type.ordinal)

    external fun unsubscribe(topic: String)
    external fun latestBytes(topic: String): ByteArray?

    /** Bytes of every sample since the last call, concatenated (queue subscriptions only). */
    external fun takeQueued(topic: String): ByteArray?

    /** One line per frame: "child parent tx ty tz qx qy qz qw age_ms". */
    external fun tfSnapshot(): String
    external fun publishPose2d(topic: String, frame: String, x: Double, y: Double, yaw: Double): Boolean

    private external fun nativeStart(domain: Int, server: String, localIp: String): String?
    private external fun nativeStop()
    private external fun nativeSubscribe(
        topic: String, type: Int, reliable: Boolean, transientLocal: Boolean, queue: Boolean,
    ): Boolean
    private external fun nativeAdvertise(topic: String, type: Int): Boolean
    external override fun publishFloat32(topic: String, v: Float): Boolean
    external override fun publishBool(topic: String, v: Boolean): Boolean
    external override fun publishString(topic: String, v: String): Boolean
    external override fun publishTwist(topic: String, vx: Double, vy: Double, wz: Double): Boolean
    external override fun latestText(topic: String): String?
    external override fun latestValues(topic: String): DoubleArray?
    external override fun latestAgeMs(topic: String): Long
    external override fun latestCount(topic: String): Long
    external override fun matchedReaders(topic: String): Int
}
