package com.quadruped.console

import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(o: Vec3) = Vec3(x + o.x, y + o.y, z + o.z)
    operator fun minus(o: Vec3) = Vec3(x - o.x, y - o.y, z - o.z)
    operator fun times(k: Double) = Vec3(x * k, y * k, z * k)
    infix fun dot(o: Vec3) = x * o.x + y * o.y + z * o.z
    infix fun cross(o: Vec3) = Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)
    fun norm() = sqrt(this dot this)
    fun unit() = this * (1.0 / norm())

    companion object {
        val ZERO = Vec3(0.0, 0.0, 0.0)
    }
}

/** Unit quaternion (x, y, z, w), as in geometry_msgs. */
data class Quat(val x: Double, val y: Double, val z: Double, val w: Double) {
    operator fun times(o: Quat) = Quat(
        w * o.x + x * o.w + y * o.z - z * o.y,
        w * o.y - x * o.z + y * o.w + z * o.x,
        w * o.z + x * o.y - y * o.x + z * o.w,
        w * o.w - x * o.x - y * o.y - z * o.z,
    )

    fun conj() = Quat(-x, -y, -z, w)

    fun rotate(v: Vec3): Vec3 {
        val u = Vec3(x, y, z)
        val t = (u cross v) * 2.0
        return v + t * w + (u cross t)
    }

    fun yaw() = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    companion object {
        val IDENTITY = Quat(0.0, 0.0, 0.0, 1.0)
        fun axisAngle(axis: Vec3, a: Double): Quat {
            val s = sin(a / 2)
            return Quat(axis.x * s, axis.y * s, axis.z * s, cos(a / 2))
        }
    }
}

/** Rigid transform: maps points in the child frame into the parent frame. */
data class Pose3(val t: Vec3, val q: Quat) {
    operator fun times(o: Pose3) = Pose3(t + q.rotate(o.t), q * o.q)
    fun apply(p: Vec3) = t + q.rotate(p)
    fun inverse(): Pose3 {
        val qi = q.conj()
        return Pose3(qi.rotate(t) * -1.0, qi)
    }

    companion object {
        val IDENTITY = Pose3(Vec3.ZERO, Quat.IDENTITY)
    }
}

/** Latest transform per child frame, from NativeLink.tfSnapshot(). Timestamps are ignored. */
class TfTree(private val edges: Map<String, Pair<String, Pose3>>) {
    private fun toRoot(frame: String): Pair<String, Pose3> {
        var f = frame
        var p = Pose3.IDENTITY
        repeat(32) {
            val (parent, t) = edges[f] ?: return f to p
            p = t * p
            f = parent
        }
        return f to p
    }

    /** Pose of `source` in `target`, or null when they are not connected. */
    fun lookup(target: String, source: String): Pose3? {
        val (rootS, s) = toRoot(source)
        val (rootT, t) = toRoot(target)
        return if (rootS == rootT) t.inverse() * s else null
    }

    fun has(frame: String) = edges.containsKey(frame)

    companion object {
        fun parse(snapshot: String): TfTree {
            val edges = HashMap<String, Pair<String, Pose3>>()
            for (line in snapshot.lineSequence()) {
                val f = line.split(' ')
                if (f.size < 9) continue
                val d = f.subList(2, 9).map { it.toDoubleOrNull() ?: Double.NaN }
                if (d.any { it.isNaN() }) continue
                edges[f[0]] = f[1] to Pose3(Vec3(d[0], d[1], d[2]), Quat(d[3], d[4], d[5], d[6]))
            }
            return TfTree(edges)
        }
    }
}
