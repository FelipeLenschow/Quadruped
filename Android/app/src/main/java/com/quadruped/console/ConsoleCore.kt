package com.quadruped.console

import org.json.JSONObject
import kotlin.math.sqrt

/** Defaults are src/quadruped_bringup/config/config.yaml (safety, control, motor). */
data class SafetyParams(
    val torquePercent: Double = 55.0,
    val rollLimitDeg: Double = 30.0,
    val pitchLimitDeg: Double = 30.0,
    val romMargin: Double = 0.15,
    val watchdogTimeout: Double = 0.35,
    val kp: Double = 25.0,
    val kd: Double = 0.5,
) {
    fun validated(): SafetyParams {
        require(torquePercent in 0.0..100.0) { "torque must be 0-100 %" }
        require(rollLimitDeg in 0.0..90.0 && pitchLimitDeg in 0.0..90.0) { "tilt limits must be 0-90 deg" }
        require(romMargin in 0.0..1.0) { "ROM margin must be 0-1" }
        require(watchdogTimeout > 0.0) { "watchdog must be > 0" }
        require(kp >= 0.0 && kd >= 0.0) { "gains must be >= 0" }
        return this
    }
}

data class Status(
    val heartbeatCount: Long,
    val heartbeatReaders: Int,
    val robotStateAgeMs: Long,
    val estopLatched: Boolean,
    val estopConfirmed: Boolean,
    val mode: String,
    val poseName: String,
    val poseProgress: Double,
    val poseLabel: String,
    val freezeBase: Boolean,
    val params: SafetyParams,
    val effectiveTorquePercent: Double,
    val maxTorqueNm: Double,
    val robotState: JSONObject?,
    val systemStats: JSONObject?,
    val systemStatsAgeMs: Long,
    val events: List<String>,
    val driveEnabled: Boolean,
    val turbo: Boolean,
    val driveCmd: DoubleArray,
)

/**
 * Port of src/quadruped_operator/quadruped_operator/console.py.
 *
 * tick() is heartbeat_loop(): call it at supervisor_frequency. Feedback topics are
 * polled there instead of arriving as callbacks; the receive counter tells a new
 * message from the last one seen.
 */
class ConsoleCore(
    private val link: RosLink,
    params: SafetyParams = SafetyParams(),
    private val motorMaxTorque: Double = 45.0,
    private val clockMs: () -> Long = System::currentTimeMillis,
) {
    companion object {
        const val HEARTBEAT = "/safety/heartbeat"
        const val MAX_TORQUE = "/safety/max_torque_percent"
        const val ROLL_LIMIT = "/safety/base_tilt_limit_deg"
        const val PITCH_LIMIT = "/safety/base_forward_tilt_limit_deg"
        const val ROM_MARGIN = "/safety/joint_rom_safety_margin"
        const val WATCHDOG = "/safety/watchdog_timeout"
        const val SAFETY_RESET = "/safety/reset"
        const val ESTOP = "/safety/estop"
        const val ESTOP_STATE = "/safety/estop_state"
        const val KP = "/control/kp"
        const val KD = "/control/kd"
        const val MODE = "/pipeline/mode"
        const val POSE_CMD = "/pose/command"
        const val POSE_INTERP = "/pose/interp_duration"
        const val FREEZE_BASE = "/base/freeze"
        const val POSE_STATUS = "/pose/status"
        const val ROBOT_STATE = "/robot_state"
        const val SYSTEM_STATS = "/system_stats"
        const val EST_VEL = "/estimator/base_lin_vel"
        const val EST_CONTACT = "/estimator/feet_contact"
        // Its own twist_mux input: priority 80, above Nav2 and skills, below the F710 and keyboard.
        const val CMD_VEL = "/cmd_vel/phone"

        // vx, vy (m/s), wz (rad/s) at full stick: joy_f710.config.yaml's scale_linear/angular.
        val DRIVE_SCALE = doubleArrayOf(0.5, 0.4, 0.5)
        val DRIVE_SCALE_TURBO = doubleArrayOf(1.0, 0.8, 1.0)

        val SUBSCRIPTIONS = listOf(
            ESTOP_STATE to MsgType.BOOL,
            ROBOT_STATE to MsgType.STRING,
            SYSTEM_STATS to MsgType.STRING,
            POSE_STATUS to MsgType.STRING,
            EST_VEL to MsgType.VECTOR3,
            EST_CONTACT to MsgType.FLOAT32_MULTI_ARRAY,
        )

        val PUBLICATIONS = listOf(
            HEARTBEAT to MsgType.FLOAT32, MAX_TORQUE to MsgType.FLOAT32, ROLL_LIMIT to MsgType.FLOAT32,
            PITCH_LIMIT to MsgType.FLOAT32, ROM_MARGIN to MsgType.FLOAT32, WATCHDOG to MsgType.FLOAT32,
            KP to MsgType.FLOAT32, KD to MsgType.FLOAT32, POSE_INTERP to MsgType.FLOAT32,
            ESTOP to MsgType.BOOL, SAFETY_RESET to MsgType.BOOL, FREEZE_BASE to MsgType.BOOL,
            MODE to MsgType.STRING, POSE_CMD to MsgType.STRING, CMD_VEL to MsgType.TWIST,
        )

        const val PREFLIGHT_MAX_SPEED = 1.5
        const val PREFLIGHT_MIN_CONTACTS = 3
        const val PREFLIGHT_MAX_AGE_MS = 1000L
        private const val MAX_EVENTS = 30
    }

    var params: SafetyParams = params.validated()
        private set
    private var estopLatched = false
    private var estopConfirmed = false
    private var mode = "pose"
    private var freezeBase = false
    private var poseName = "none"
    private var poseProgress = 0.0
    private var poseLabel = ""
    private var robotState: JSONObject? = null
    private var systemStats: JSONObject? = null
    private var heartbeatCount = 0L
    private val seen = HashMap<String, Long>()
    private val events = ArrayDeque<String>()
    private var driveEnabled = false
    private var turbo = false
    private val stick = DoubleArray(3)
    private var robotEstop: Boolean? = null
    private var robotEstopStreak = 0

    @Synchronized
    fun tick() {
        pollFeedback()

        // Keep asserting the kill until the robot echoes it back (console.py).
        if (estopLatched && !estopConfirmed) link.publishBool(ESTOP, true)

        link.publishFloat32(HEARTBEAT, (clockMs() / 1000.0).toFloat())
        link.publishFloat32(MAX_TORQUE, if (estopLatched) 0f else params.torquePercent.toFloat())
        link.publishFloat32(ROLL_LIMIT, params.rollLimitDeg.toFloat())
        link.publishFloat32(PITCH_LIMIT, params.pitchLimitDeg.toFloat())
        link.publishFloat32(ROM_MARGIN, params.romMargin.toFloat())
        link.publishFloat32(WATCHDOG, params.watchdogTimeout.toFloat())
        link.publishFloat32(KP, params.kp.toFloat())
        link.publishFloat32(KD, params.kd.toFloat())
        heartbeatCount++
    }

    private fun fresh(topic: String): Boolean {
        val n = link.latestCount(topic)
        if (n == 0L || n == seen[topic]) return false
        seen[topic] = n
        return true
    }

    private fun pollFeedback() {
        if (fresh(ESTOP_STATE)) onEstopState((link.latestValues(ESTOP_STATE)?.firstOrNull() ?: 0.0) > 0.5)
        if (fresh(ROBOT_STATE)) {
            try {
                val js = JSONObject(link.latestText(ROBOT_STATE) ?: "")
                robotState = js
                mode = js.optString("mode", mode)
                if (js.has("estop")) onRobotStateEstop(js.getBoolean("estop"))
            } catch (_: Exception) {
            }
        }
        if (fresh(SYSTEM_STATS)) {
            try {
                systemStats = JSONObject(link.latestText(SYSTEM_STATS) ?: "")
            } catch (_: Exception) {
            }
        }
        if (fresh(POSE_STATUS)) {
            val parts = (link.latestText(POSE_STATUS) ?: "").split("|")
            if (parts.size >= 3) {
                parts[1].toDoubleOrNull()?.let {
                    poseName = parts[0]
                    poseProgress = it
                    poseLabel = parts[2]
                }
            }
        }
    }

    /** The robot owns the latch: we set it, only the robot releases it. */
    private fun onEstopState(latched: Boolean) {
        estopConfirmed = latched
        if (latched) stopDrive()
        if (latched == estopLatched) return
        estopLatched = latched
        if (!latched) log("E-STOP released at the robot. Robot is in POSE mode: pick a pose to stand.")
    }

    /**
     * /robot_state's "estop" (10 Hz) backs up /safety/estop_state, which the robot sends
     * once per change. Losing that one ack left the phone re-asserting /safety/estop,
     * which re-latched the robot the moment ENTER released it. Acted on only after two
     * messages agree, since one can predate the robot handling our stop or its release.
     */
    private fun onRobotStateEstop(latched: Boolean) {
        robotEstopStreak = if (latched == robotEstop) robotEstopStreak + 1 else 1
        robotEstop = latched
        if (robotEstopStreak != 2) return
        if (latched) {
            estopConfirmed = true
            if (!estopLatched) {
                estopLatched = true
                stopDrive()
                log("E-STOP latched on the robot.")
            }
        } else if (estopLatched && estopConfirmed) {
            estopLatched = false
            estopConfirmed = false
            log("E-STOP released at the robot. Robot is in POSE mode: pick a pose to stand.")
        }
    }

    @Synchronized
    fun triggerEstop(source: String) {
        if (estopLatched) return
        estopLatched = true
        estopConfirmed = false
        stopDrive()
        link.publishBool(ESTOP, true)
        link.publishFloat32(MAX_TORQUE, 0f)
        mode = "pose"
        link.publishString(MODE, "pose")
        log("E-STOP by $source: torque 0, mode POSE. Release with ENTER on the driver.")
    }

    /** For shutdown: the caller should keep the link up ~100 ms afterwards so it flushes. */
    @Synchronized
    fun emitEstop() {
        repeat(3) { link.publishBool(ESTOP, true) }
        link.publishFloat32(MAX_TORQUE, 0f)
    }

    @Synchronized
    fun safetyReset() {
        link.publishBool(SAFETY_RESET, true)
        log("Safety reset sent. If the cause is still there it latches again.")
    }

    @Synchronized
    fun preflight(): List<String> {
        val vel = link.latestValues(EST_VEL) ?: return listOf("no estimator data (is the driver running?)")
        val reasons = mutableListOf<String>()
        val age = link.latestAgeMs(EST_VEL)
        if (age > PREFLIGHT_MAX_AGE_MS) reasons += "estimator data is stale (${age / 1000.0} s old)"
        val speed = sqrt(vel.sumOf { it * it })
        if (speed > PREFLIGHT_MAX_SPEED) {
            reasons += "base_lin_vel = ${"%.1f".format(speed)} m/s (limit $PREFLIGHT_MAX_SPEED) - estimator has diverged"
        }
        link.latestValues(EST_CONTACT)?.let { c ->
            val n = c.count { it > 0.5 }
            if (n < PREFLIGHT_MIN_CONTACTS) reasons += "only $n/4 feet in contact"
        }
        return reasons
    }

    /** Returns why policy mode was refused; empty when the mode was sent. */
    @Synchronized
    fun setMode(newMode: String, force: Boolean = false): List<String> {
        require(newMode == "pose" || newMode == "policy")
        if (newMode == "policy" && !force) {
            val reasons = preflight()
            if (reasons.isNotEmpty()) {
                log("POLICY REFUSED: " + reasons.joinToString("; "))
                return reasons
            }
        }
        mode = newMode
        link.publishString(MODE, newMode)
        log("Mode = $newMode${if (force) " (forced)" else ""}")
        return emptyList()
    }

    @Synchronized
    fun sendPose(name: String) {
        link.publishString(POSE_CMD, name)
        if (mode != "pose") {
            mode = "pose"
            link.publishString(MODE, "pose")
        }
        log("Pose = $name")
    }

    @Synchronized
    fun setInterp(seconds: Double) {
        if (seconds <= 0.0) return
        link.publishFloat32(POSE_INTERP, seconds.toFloat())
        log("Interp = $seconds s")
    }

    @Synchronized
    fun setFreezeBase(on: Boolean) {
        freezeBase = on
        link.publishBool(FREEZE_BASE, on)
        log("Freeze Base = ${if (on) "on" else "off"}")
    }

    @Synchronized
    fun updateParams(p: SafetyParams) {
        val old = params
        params = p.validated()
        if (p.kp != old.kp) link.publishFloat32(KP, p.kp.toFloat())
        if (p.kd != old.kd) link.publishFloat32(KD, p.kd.toFloat())
        log("Params: $p")
    }

    /**
     * Phone drive on CMD_VEL (twist_mux priority above Nav2). While on, a command
     * goes out every tickDrive() - zero with the sticks centred - so it holds Nav2 off.
     * Off sends one zero and then nothing, and twist_mux hands back after its timeout.
     */
    @Synchronized
    fun setDrive(on: Boolean) {
        if (on == driveEnabled) return
        if (on && estopLatched) {
            log("Drive not enabled: e-stop is latched")
            return
        }
        if (on) {
            stick.fill(0.0)
            driveEnabled = true
            log("Phone drive ON (overrides Nav2)")
        } else {
            stopDrive()
        }
    }

    private fun stopDrive() {
        if (!driveEnabled) return
        driveEnabled = false
        stick.fill(0.0)
        link.publishTwist(CMD_VEL, 0.0, 0.0, 0.0)
        log("Phone drive OFF")
    }

    @Synchronized
    fun setTurbo(on: Boolean) {
        turbo = on
    }

    /** Stick deflections in -1..1: forward, left, turn left (counter-clockwise). */
    @Synchronized
    fun setMove(forward: Double, left: Double) {
        if (!driveEnabled) return
        stick[0] = forward.coerceIn(-1.0, 1.0)
        stick[1] = left.coerceIn(-1.0, 1.0)
    }

    @Synchronized
    fun setTurn(left: Double) {
        if (!driveEnabled) return
        stick[2] = left.coerceIn(-1.0, 1.0)
    }

    private fun driveCmd(): DoubleArray {
        val k = if (turbo) DRIVE_SCALE_TURBO else DRIVE_SCALE
        return DoubleArray(3) { stick[it] * k[it] }
    }

    @Synchronized
    fun tickDrive() {
        if (!driveEnabled) return
        val c = driveCmd()
        link.publishTwist(CMD_VEL, c[0], c[1], c[2])
    }

    @Synchronized
    fun status(): Status {
        val torque = if (estopLatched) 0.0 else params.torquePercent
        return Status(
            heartbeatCount = heartbeatCount,
            heartbeatReaders = link.matchedReaders(HEARTBEAT),
            robotStateAgeMs = link.latestAgeMs(ROBOT_STATE),
            estopLatched = estopLatched,
            estopConfirmed = estopConfirmed,
            mode = mode,
            poseName = poseName,
            poseProgress = poseProgress,
            poseLabel = poseLabel,
            freezeBase = freezeBase,
            params = params,
            effectiveTorquePercent = torque,
            maxTorqueNm = torque / 100.0 * motorMaxTorque,
            robotState = robotState,
            systemStats = systemStats,
            systemStatsAgeMs = link.latestAgeMs(SYSTEM_STATS),
            events = events.toList(),
            driveEnabled = driveEnabled,
            turbo = turbo,
            driveCmd = driveCmd(),
        )
    }

    private fun log(msg: String) {
        events.addFirst(msg)
        while (events.size > MAX_EVENTS) events.removeLast()
    }
}
