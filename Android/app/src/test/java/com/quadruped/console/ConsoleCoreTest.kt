package com.quadruped.console

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

private class FakeLink : RosLink {
    val sent = mutableListOf<Pair<String, Any>>()
    val text = HashMap<String, String>()
    val values = HashMap<String, DoubleArray>()
    val counts = HashMap<String, Long>()
    val ages = HashMap<String, Long>()

    fun receive(topic: String, v: DoubleArray) {
        values[topic] = v
        counts[topic] = (counts[topic] ?: 0) + 1
        ages[topic] = 0
    }

    fun receive(topic: String, s: String) {
        text[topic] = s
        counts[topic] = (counts[topic] ?: 0) + 1
        ages[topic] = 0
    }

    fun on(topic: String) = sent.filter { it.first == topic }.map { it.second }

    override fun publishFloat32(topic: String, v: Float) = sent.add(topic to v)
    override fun publishBool(topic: String, v: Boolean) = sent.add(topic to v)
    override fun publishString(topic: String, v: String) = sent.add(topic to v)
    override fun publishTwist(topic: String, vx: Double, vy: Double, wz: Double) = sent.add(topic to listOf(vx, vy, wz))
    override fun latestText(topic: String) = text[topic]
    override fun latestValues(topic: String) = values[topic]
    override fun latestAgeMs(topic: String) = ages[topic] ?: -1
    override fun latestCount(topic: String) = counts[topic] ?: 0
    override fun matchedReaders(topic: String) = 1
}

class ConsoleCoreTest {
    private val link = FakeLink()
    private val core = ConsoleCore(link)

    @Test
    fun tickPublishesHeartbeatAndParams() {
        core.tick()
        assertEquals(1, link.on(ConsoleCore.HEARTBEAT).size)
        assertEquals(listOf(55f), link.on(ConsoleCore.MAX_TORQUE))
        assertEquals(listOf(0.15f), link.on(ConsoleCore.ROM_MARGIN))
        assertEquals(listOf(0.35f), link.on(ConsoleCore.WATCHDOG))
        assertEquals(listOf(25f), link.on(ConsoleCore.KP))
        assertTrue(link.on(ConsoleCore.ESTOP).isEmpty())
    }

    @Test
    fun estopCutsTorqueAndRepeatsUntilAcked() {
        core.triggerEstop("test")
        assertEquals(listOf(true), link.on(ConsoleCore.ESTOP))
        assertEquals(listOf(0f), link.on(ConsoleCore.MAX_TORQUE))
        assertEquals(listOf("pose"), link.on(ConsoleCore.MODE))

        link.sent.clear()
        core.tick()
        core.tick()
        assertEquals(listOf(true, true), link.on(ConsoleCore.ESTOP))
        assertEquals(listOf(0f, 0f), link.on(ConsoleCore.MAX_TORQUE))
        assertEquals(2, link.on(ConsoleCore.HEARTBEAT).size)

        link.receive(ConsoleCore.ESTOP_STATE, doubleArrayOf(1.0))
        link.sent.clear()
        core.tick()
        assertTrue(link.on(ConsoleCore.ESTOP).isEmpty())
        assertEquals(listOf(0f), link.on(ConsoleCore.MAX_TORQUE))
        assertTrue(core.status().estopConfirmed)
    }

    @Test
    fun releaseAtRobotRestoresTorque() {
        core.triggerEstop("test")
        link.receive(ConsoleCore.ESTOP_STATE, doubleArrayOf(1.0))
        core.tick()
        link.receive(ConsoleCore.ESTOP_STATE, doubleArrayOf(0.0))
        link.sent.clear()
        core.tick()
        assertFalse(core.status().estopLatched)
        assertEquals(listOf(55f), link.on(ConsoleCore.MAX_TORQUE))
    }

    @Test
    fun estopLatchedElsewhereIsFollowed() {
        link.receive(ConsoleCore.ESTOP_STATE, doubleArrayOf(1.0))
        core.tick()
        assertTrue(core.status().estopLatched)
        assertEquals(listOf(0f), link.on(ConsoleCore.MAX_TORQUE))
        assertTrue(link.on(ConsoleCore.ESTOP).isEmpty())
    }

    @Test
    fun policyRefusedWithoutEstimator() {
        val reasons = core.setMode("policy")
        assertEquals(1, reasons.size)
        assertTrue(link.on(ConsoleCore.MODE).isEmpty())
        assertTrue(core.setMode("policy", force = true).isEmpty())
        assertEquals(listOf("policy"), link.on(ConsoleCore.MODE))
    }

    @Test
    fun policyPreflightChecksSpeedContactsAndAge() {
        link.receive(ConsoleCore.EST_VEL, doubleArrayOf(0.0, 0.0, -49.0))
        link.receive(ConsoleCore.EST_CONTACT, doubleArrayOf(1.0, 0.0, 0.0, 1.0))
        link.ages[ConsoleCore.EST_VEL] = 2000
        assertEquals(3, core.preflight().size)

        link.receive(ConsoleCore.EST_VEL, doubleArrayOf(0.01, 0.0, 0.0))
        link.receive(ConsoleCore.EST_CONTACT, doubleArrayOf(1.0, 1.0, 1.0, 0.0))
        assertTrue(core.setMode("policy").isEmpty())
    }

    @Test
    fun poseSwitchesBackToPoseMode() {
        core.setMode("policy", force = true)
        link.sent.clear()
        core.sendPose("stand")
        assertEquals(listOf("stand"), link.on(ConsoleCore.POSE_CMD))
        assertEquals(listOf("pose"), link.on(ConsoleCore.MODE))
    }

    @Test
    fun robotStateAndPoseStatusAreParsed() {
        link.receive(ConsoleCore.ROBOT_STATE, """{"mode": "policy", "posture": "walking"}""")
        link.receive(ConsoleCore.POSE_STATUS, "stand|0.5|interpolating")
        core.tick()
        val s = core.status()
        assertEquals("policy", s.mode)
        assertEquals("walking", s.robotState!!.getString("posture"))
        assertEquals("stand", s.poseName)
        assertEquals(0.5, s.poseProgress, 1e-9)
    }

    @Test
    fun emitEstopSendsThreeKillsAndZeroTorque() {
        core.emitEstop()
        assertEquals(listOf(true, true, true), link.on(ConsoleCore.ESTOP))
        assertEquals(listOf(0f), link.on(ConsoleCore.MAX_TORQUE))
    }

    @Test(expected = IllegalArgumentException::class)
    fun badParamsRejected() {
        core.updateParams(SafetyParams(torquePercent = 150.0))
    }

    @Test
    fun driveOffPublishesNothing() {
        core.setMove(1.0, 0.0)
        core.tickDrive()
        assertTrue(link.on(ConsoleCore.CMD_VEL).isEmpty())
    }

    @Test
    fun driveOnHoldsZeroThenScalesSticks() {
        core.setDrive(true)
        core.tickDrive()
        assertEquals(listOf(listOf(0.0, 0.0, 0.0)), link.on(ConsoleCore.CMD_VEL))
        link.sent.clear()
        core.setMove(1.0, -2.0)
        core.setTurn(0.5)
        core.tickDrive()
        assertEquals(listOf(listOf(0.5, -0.4, 0.25)), link.on(ConsoleCore.CMD_VEL))
        link.sent.clear()
        core.setTurbo(true)
        core.tickDrive()
        assertEquals(listOf(listOf(1.0, -0.8, 0.5)), link.on(ConsoleCore.CMD_VEL))
    }

    @Test
    fun driveOffSendsOneZeroThenStops() {
        core.setDrive(true)
        core.setMove(1.0, 0.0)
        link.sent.clear()
        core.setDrive(false)
        core.tickDrive()
        core.tickDrive()
        assertEquals(listOf(listOf(0.0, 0.0, 0.0)), link.on(ConsoleCore.CMD_VEL))
    }

    @Test
    fun estopTurnsDriveOffAndBlocksIt() {
        core.setDrive(true)
        core.triggerEstop("test")
        assertFalse(core.status().driveEnabled)
        core.setDrive(true)
        assertFalse(core.status().driveEnabled)
    }

    @Test
    fun estopLatchedAtRobotTurnsDriveOff() {
        core.setDrive(true)
        link.receive(ConsoleCore.ESTOP_STATE, doubleArrayOf(1.0))
        core.tick()
        assertFalse(core.status().driveEnabled)
    }

    private fun robotState(estop: Boolean) {
        link.receive(ConsoleCore.ROBOT_STATE, """{"mode": "pose", "estop": $estop}""")
        core.tick()
    }

    @Test
    fun lostAckIsCoveredByRobotStateSoEnterCanRelease() {
        core.triggerEstop("test")
        robotState(true)
        robotState(true)
        assertTrue(core.status().estopConfirmed)
        link.sent.clear()
        core.tick()
        assertTrue("must stop re-asserting once the robot shows the latch", link.on(ConsoleCore.ESTOP).isEmpty())

        robotState(false)
        robotState(false)
        assertFalse(core.status().estopLatched)
        link.sent.clear()
        core.tick()
        assertTrue(link.on(ConsoleCore.ESTOP).isEmpty())
        assertEquals(listOf(55f), link.on(ConsoleCore.MAX_TORQUE))
    }

    @Test
    fun staleRobotStateDoesNotReleaseAnUnackedStop() {
        core.triggerEstop("test")
        robotState(false)
        robotState(false)
        assertTrue(core.status().estopLatched)
        link.sent.clear()
        core.tick()
        assertEquals(listOf(true), link.on(ConsoleCore.ESTOP))
    }

    @Test
    fun singleRobotStateMessageIsNotEnough() {
        core.triggerEstop("test")
        robotState(true)
        assertFalse(core.status().estopConfirmed)
        robotState(true)
        robotState(false)
        assertTrue(core.status().estopLatched)
    }

    @Test
    fun robotLatchedElsewhereIsAdoptedWithoutReasserting() {
        robotState(true)
        robotState(true)
        assertTrue(core.status().estopLatched)
        link.sent.clear()
        core.tick()
        assertTrue(link.on(ConsoleCore.ESTOP).isEmpty())
        assertEquals(listOf(0f), link.on(ConsoleCore.MAX_TORQUE))
    }

    @Test
    fun systemStatsAreParsed() {
        link.receive(ConsoleCore.SYSTEM_STATS, """{"cpu": 45.2, "cores": [40.0, 50.0], "gpu": null, "ram_used_mb": 3100}""")
        core.tick()
        val st = core.status().systemStats!!
        assertEquals(45.2, st.getDouble("cpu"), 1e-9)
        assertTrue(st.isNull("gpu"))
        assertEquals(2, st.getJSONArray("cores").length())
    }
}
