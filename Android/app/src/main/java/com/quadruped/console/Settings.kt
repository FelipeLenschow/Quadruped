package com.quadruped.console

import android.content.Context

/** Network defaults are config.yaml's `network` block. */
data class LinkSettings(
    val domainId: Int = 42,
    val discoveryMode: String = "auto", // auto | on | off, as network.discovery_mode
    val discoveryServer: String = "10.42.0.1:11811",
    val heartbeatHz: Double = 10.0,
)

object Settings {
    private const val NAME = "console"

    fun loadLink(ctx: Context): LinkSettings {
        val p = ctx.getSharedPreferences(NAME, Context.MODE_PRIVATE)
        val d = LinkSettings()
        return LinkSettings(
            domainId = p.getInt("domain", d.domainId),
            discoveryMode = p.getString("discovery_mode", d.discoveryMode)!!,
            discoveryServer = p.getString("discovery_server", d.discoveryServer)!!,
            heartbeatHz = p.getFloat("heartbeat_hz", d.heartbeatHz.toFloat()).toDouble(),
        )
    }

    fun saveLink(ctx: Context, s: LinkSettings) {
        ctx.getSharedPreferences(NAME, Context.MODE_PRIVATE).edit()
            .putInt("domain", s.domainId)
            .putString("discovery_mode", s.discoveryMode)
            .putString("discovery_server", s.discoveryServer)
            .putFloat("heartbeat_hz", s.heartbeatHz.toFloat())
            .apply()
    }

    fun loadParams(ctx: Context): SafetyParams {
        val p = ctx.getSharedPreferences(NAME, Context.MODE_PRIVATE)
        val d = SafetyParams()
        fun f(k: String, v: Double) = p.getFloat(k, v.toFloat()).toDouble()
        return SafetyParams(
            torquePercent = f("torque", d.torquePercent),
            rollLimitDeg = f("roll", d.rollLimitDeg),
            pitchLimitDeg = f("pitch", d.pitchLimitDeg),
            romMargin = f("rom", d.romMargin),
            watchdogTimeout = f("watchdog", d.watchdogTimeout),
            kp = f("kp", d.kp),
            kd = f("kd", d.kd),
        )
    }

    fun saveParams(ctx: Context, s: SafetyParams) {
        ctx.getSharedPreferences(NAME, Context.MODE_PRIVATE).edit()
            .putFloat("torque", s.torquePercent.toFloat())
            .putFloat("roll", s.rollLimitDeg.toFloat())
            .putFloat("pitch", s.pitchLimitDeg.toFloat())
            .putFloat("rom", s.romMargin.toFloat())
            .putFloat("watchdog", s.watchdogTimeout.toFloat())
            .putFloat("kp", s.kp.toFloat())
            .putFloat("kd", s.kd.toFloat())
            .apply()
    }
}

/** The robot's Jetson over SSH. Shortcuts are "Label = command" lines; tapping one types it, without Enter. */
data class ShellSettings(
    val host: String = "10.42.0.1",
    val port: Int = 22,
    val user: String = "unitree",
    val shortcuts: String = DEFAULT_SHORTCUTS,
) {
    fun shortcutList(): List<Pair<String, String>> = shortcuts.lines().mapNotNull { line ->
        val i = line.indexOf('=')
        if (i <= 0) null else line.substring(0, i).trim() to line.substring(i + 1).trim()
    }

    companion object {
        val DEFAULT_SHORTCUTS = """
            tmux = tmux new -A -s robot \; set -g mouse on
            container = cd ~/Quadruped && quaddocker
            attach = quadattach
            disc = fast-discovery-server -i 0 -l 10.42.0.1 -p 11811
            launcher = python3 launcher.py
            real = ros2 launch quadruped_bringup real.launch.py checkpoint:=
            nav = ros2 launch quadruped_bringup nav.launch.py sim:=false
        """.trimIndent()

        fun load(ctx: Context): ShellSettings {
            val p = ctx.getSharedPreferences("console", Context.MODE_PRIVATE)
            val d = ShellSettings()
            return ShellSettings(
                host = p.getString("ssh_host", d.host)!!,
                port = p.getInt("ssh_port", d.port),
                user = p.getString("ssh_user", d.user)!!,
                shortcuts = p.getString("ssh_shortcuts", d.shortcuts)!!
                    .replace(Regex("(?m)^tmux = tmux new -A -s robot$"), "tmux = tmux new -A -s robot \\\\; set -g mouse on"),
            )
        }

        fun save(ctx: Context, s: ShellSettings) {
            ctx.getSharedPreferences("console", Context.MODE_PRIVATE).edit()
                .putString("ssh_host", s.host)
                .putInt("ssh_port", s.port)
                .putString("ssh_user", s.user)
                .putString("ssh_shortcuts", s.shortcuts)
                .apply()
        }
    }
}
