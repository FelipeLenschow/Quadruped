package com.quadruped.console

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class GeometryTest {
    private fun near(a: Vec3, b: Vec3) {
        assertEquals(a.x, b.x, 1e-6)
        assertEquals(a.y, b.y, 1e-6)
        assertEquals(a.z, b.z, 1e-6)
    }

    private val urdf = File("src/main/assets/go2/go2.urdf").inputStream().use { Urdf.parse(it) }

    @Test
    fun urdfRootAndMeshes() {
        assertEquals("base", urdf.root)
        assertEquals("base", urdf.visuals.getValue("base").single().mesh)
        assertEquals("thigh_mirror", urdf.visuals.getValue("FR_thigh").single().mesh)
    }

    @Test
    fun straightLegHangsBelowThighJoint() {
        val foot = urdf.linkPoses(emptyMap()).getValue("FL_foot").t
        near(Vec3(0.1934, 0.142, -0.426), foot)
    }

    @Test
    fun standPoseFootIsUnderHipAtStandingHeight() {
        // config.yaml stand: thigh 0.8, calf -1.5
        val q = mapOf("RR_hip_joint" to -0.1, "RR_thigh_joint" to 0.8, "RR_calf_joint" to -1.5)
        val foot = urdf.linkPoses(q).getValue("RR_foot").t
        assertEquals(-0.1934, foot.x, 0.03)
        assertEquals(-0.31, foot.z, 0.02)
    }

    @Test
    fun hipAbductionSwingsFootOutward() {
        val foot = urdf.linkPoses(mapOf("FL_hip_joint" to 0.3)).getValue("FL_foot").t
        val straight = urdf.linkPoses(emptyMap()).getValue("FL_foot").t
        assertTrue(foot.y > straight.y)
    }

    @Test
    fun tfChainsThroughParents() {
        val yaw = PI / 2
        val tree = TfTree.parse(
            "odom map 1 0 0 0 0 ${sin(yaw / 2)} ${cos(yaw / 2)} 5\n" +
                "base_footprint odom 2 0 0 0 0 0 1 3\n",
        )
        val p = tree.lookup("map", "base_footprint")!!
        near(Vec3(1.0, 2.0, 0.0), p.t)
        assertEquals(yaw, p.q.yaw(), 1e-9)
        near(Vec3.ZERO, tree.lookup("base_footprint", "base_footprint")!!.t)
        near(Vec3(-2.0, 0.0, 0.0), tree.lookup("base_footprint", "odom")!!.t)
        assertNull(tree.lookup("map", "camera"))
    }
}
