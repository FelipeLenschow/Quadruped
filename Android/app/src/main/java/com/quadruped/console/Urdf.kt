package com.quadruped.console

import org.w3c.dom.Element
import java.io.InputStream
import javax.xml.parsers.DocumentBuilderFactory

/** The parts of a URDF needed to pose and draw it: joints, and visual meshes per link. */
class Urdf(val root: String, val joints: List<Joint>, val visuals: Map<String, List<Visual>>) {
    data class Joint(
        val name: String, val type: String, val parent: String, val child: String,
        val origin: Pose3, val axis: Vec3,
    )

    /** `mesh` is the file's base name, e.g. "thigh_mirror" for package://.../thigh_mirror.dae. */
    data class Visual(val mesh: String, val origin: Pose3)

    /** Children before grandchildren, so a parent's pose is always ready. */
    private val ordered: List<Joint> = run {
        val byParent = joints.groupBy { it.parent }
        val out = mutableListOf<Joint>()
        val queue = ArrayDeque(listOf(root))
        while (queue.isNotEmpty()) {
            byParent[queue.removeFirst()]?.forEach {
                out += it
                queue += it.child
            }
        }
        out
    }

    /** Every link's pose in the root link's frame. Joints missing from `q` sit at zero. */
    fun linkPoses(q: Map<String, Double>): Map<String, Pose3> {
        val poses = HashMap<String, Pose3>()
        poses[root] = Pose3.IDENTITY
        for (j in ordered) {
            val parent = poses[j.parent] ?: continue
            val a = q[j.name] ?: 0.0
            val motion = when (j.type) {
                "revolute", "continuous" -> Pose3(Vec3.ZERO, Quat.axisAngle(j.axis, a))
                "prismatic" -> Pose3(j.axis * a, Quat.IDENTITY)
                else -> Pose3.IDENTITY
            }
            poses[j.child] = parent * j.origin * motion
        }
        return poses
    }

    companion object {
        fun parse(input: InputStream): Urdf {
            val doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(input)
            val robot = doc.documentElement
            val joints = robot.children("joint").map { j ->
                Joint(
                    name = j.getAttribute("name"),
                    type = j.getAttribute("type"),
                    parent = j.children("parent").first().getAttribute("link"),
                    child = j.children("child").first().getAttribute("link"),
                    origin = origin(j.children("origin").firstOrNull()),
                    axis = j.children("axis").firstOrNull()?.getAttribute("xyz")?.let(::vec)
                        ?.takeIf { it.norm() > 0 }?.unit() ?: Vec3(1.0, 0.0, 0.0),
                )
            }
            val visuals = robot.children("link").associate { link ->
                link.getAttribute("name") to link.children("visual").mapNotNull { v ->
                    val mesh = v.children("geometry").firstOrNull()?.children("mesh")?.firstOrNull()
                        ?: return@mapNotNull null
                    val file = mesh.getAttribute("filename").substringAfterLast('/').substringBeforeLast('.')
                    Visual(file, origin(v.children("origin").firstOrNull()))
                }
            }.filterValues { it.isNotEmpty() }
            val children = joints.map { it.child }.toSet()
            val root = robot.children("link").map { it.getAttribute("name") }.first { it !in children }
            return Urdf(root, joints, visuals)
        }

        private fun Element.children(tag: String): List<Element> {
            val out = mutableListOf<Element>()
            val nodes = childNodes
            for (i in 0 until nodes.length) {
                val n = nodes.item(i)
                if (n is Element && n.tagName == tag) out += n
            }
            return out
        }

        private fun vec(s: String): Vec3 {
            val v = s.trim().split(Regex("\\s+")).map { it.toDouble() }
            return Vec3(v[0], v[1], v[2])
        }

        /** URDF rpy is fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll). */
        private fun origin(e: Element?): Pose3 {
            if (e == null) return Pose3.IDENTITY
            val xyz = e.getAttribute("xyz").takeIf { it.isNotBlank() }?.let(::vec) ?: Vec3.ZERO
            val rpy = e.getAttribute("rpy").takeIf { it.isNotBlank() }?.let(::vec) ?: Vec3.ZERO
            val q = Quat.axisAngle(Vec3(0.0, 0.0, 1.0), rpy.z) *
                Quat.axisAngle(Vec3(0.0, 1.0, 0.0), rpy.y) *
                Quat.axisAngle(Vec3(1.0, 0.0, 0.0), rpy.x)
            return Pose3(xyz, q)
        }
    }
}
