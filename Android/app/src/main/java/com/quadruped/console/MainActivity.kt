package com.quadruped.console

import android.Manifest
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quadruped.console.ui.ConsoleTheme
import com.quadruped.console.ui.Joystick
import com.quadruped.console.ui.MapView
import com.quadruped.console.ui.Robot3D
import com.quadruped.console.ui.Mono
import com.quadruped.console.ui.Palette
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= 33) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 0)
        setContent { ConsoleTheme { ConsoleApp() } }
    }

    /** Nobody is holding the sticks once the screen is gone. */
    override fun onPause() {
        ConsoleService.core?.setDrive(false)
        super.onPause()
    }
}

private enum class Tab(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    DRIVE("Drive", Icons.Filled.PlayArrow),
    ROBOT("Robot", Icons.Filled.Face),
    TUNE("Tune", Icons.Filled.Build),
    MAP("Map", Icons.Filled.LocationOn),
    LINK("Link", Icons.Filled.Settings),
}

@Composable
fun ConsoleApp() {
    var status by remember { mutableStateOf<Status?>(null) }
    var info by remember { mutableStateOf("") }
    var tab by remember { mutableIntStateOf(0) }
    var core by remember { mutableStateOf<ConsoleCore?>(null) }
    LaunchedEffect(Unit) {
        while (true) {
            core = ConsoleService.core
            status = core?.status()
            info = ConsoleService.linkInfo
            delay(100)
        }
    }

    Scaffold(
        containerColor = Palette.Bg,
        bottomBar = {
            NavigationBar(containerColor = Palette.Surface) {
                Tab.entries.forEachIndexed { i, t ->
                    NavigationBarItem(
                        selected = tab == i, onClick = { tab = i },
                        icon = { Icon(t.icon, t.label) }, label = { Text(t.label) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = Palette.Bg, indicatorColor = Palette.Accent,
                            unselectedIconColor = Palette.Muted, unselectedTextColor = Palette.Muted,
                            selectedTextColor = Palette.Accent,
                        ),
                    )
                }
            }
        },
    ) { pad ->
        Column(
            Modifier.fillMaxSize().padding(bottom = pad.calculateBottomPadding()).statusBarsPadding()
                .padding(horizontal = 16.dp),
        ) {
            TopBar(status, core != null)
            Spacer(Modifier.height(12.dp))
            EstopButton(status, core)
            Spacer(Modifier.height(12.dp))
            Column(
                Modifier.weight(1f).verticalScroll(
                    rememberScrollState(),
                    // Sticks, the 3D view and the map take drags themselves.
                    enabled = Tab.entries[tab] != Tab.DRIVE && Tab.entries[tab] != Tab.MAP &&
                        !(Tab.entries[tab] == Tab.ROBOT && status?.mode == "policy"),
                ),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                val s = status
                val c = core
                when (Tab.entries[tab]) {
                    Tab.LINK -> LinkTab(s, info)
                    Tab.TUNE -> TuneTab(s, c)
                    else -> if (s == null || c == null) OfflineCard(info) else when (Tab.entries[tab]) {
                        Tab.DRIVE -> DriveTab(s, c)
                        Tab.MAP -> MapView(Modifier.fillMaxWidth().height(520.dp))
                        else -> RobotTab(s, c)
                    }
                }
                Spacer(Modifier.height(8.dp))
            }
        }
    }
}

// ---------------------------------------------------------------- chrome

@Composable
private fun TopBar(s: Status?, connected: Boolean) {
    val ctx = LocalContext.current
    Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("QUADRUPED", style = MaterialTheme.typography.labelSmall, color = Palette.Muted)
            Text("Console", style = MaterialTheme.typography.titleLarge)
        }
        LinkPill(s, connected)
        Spacer(Modifier.width(8.dp))
        if (connected) {
            OutlinedButton(
                { ConsoleService.stop(ctx) }, border = BorderStroke(1.dp, Palette.Outline),
                contentPadding = PaddingValues(horizontal = 12.dp),
            ) { Text("Disconnect", color = Palette.Muted, fontSize = 13.sp) }
        } else {
            Button(
                { ConsoleService.start(ctx) },
                colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.Bg),
            ) { Text("Connect", fontWeight = FontWeight.Bold) }
        }
    }
}

@Composable
private fun LinkPill(s: Status?, connected: Boolean) {
    val (color, text) = when {
        !connected || s == null -> Palette.Muted to "Offline"
        s.robotStateAgeMs < 0 -> Palette.Warn to "Waiting"
        s.robotStateAgeMs > 1000 -> Palette.Danger to "Lost ${s.robotStateAgeMs / 1000}s"
        else -> Palette.Ok to "Live"
    }
    Row(
        Modifier.clip(CircleShape).background(color.copy(alpha = 0.12f)).padding(horizontal = 10.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(color))
        Spacer(Modifier.width(6.dp))
        Text(text, color = color, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
private fun EstopButton(s: Status?, core: ConsoleCore?) {
    val haptic = LocalHapticFeedback.current
    val latched = s?.estopLatched == true
    val pulse by rememberInfiniteTransition(label = "pulse").animateFloat(
        0.35f, 1f, infiniteRepeatable(tween(700), RepeatMode.Reverse), label = "a",
    )
    val shape = RoundedCornerShape(20.dp)
    val enabled = core != null
    Box(
        Modifier.fillMaxWidth().height(if (latched) 96.dp else 84.dp).clip(shape)
            .background(
                if (!enabled) Brush.verticalGradient(listOf(Palette.Surface2, Palette.Surface))
                else if (latched) Brush.verticalGradient(listOf(Color(0xFF3A0D14), Color(0xFF240810)))
                else Brush.verticalGradient(listOf(Palette.Danger, Palette.DangerDeep)),
            )
            .border(2.dp, if (latched) Palette.Danger.copy(alpha = pulse) else Color.Transparent, shape)
            .clickable(enabled = enabled) {
                haptic.performHapticFeedback(HapticFeedbackType.LongPress)
                core?.triggerEstop("phone button")
            },
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                if (latched) "E-STOP LATCHED" else "STOP",
                color = if (enabled) Color.White else Palette.Muted,
                fontSize = if (latched) 24.sp else 34.sp, fontWeight = FontWeight.Black, letterSpacing = 4.sp,
            )
            if (latched) {
                Text(
                    if (s?.estopConfirmed == true) "Release with ENTER on the robot's driver"
                    else "Waiting for the robot to confirm…",
                    color = Palette.Danger, fontSize = 12.sp,
                )
            }
        }
    }
}

@Composable
private fun Card(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) =
    Column(
        modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Palette.Surface)
            .border(1.dp, Palette.Outline.copy(alpha = 0.6f), RoundedCornerShape(18.dp)).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp), content = content,
    )

@Composable
private fun SectionLabel(text: String) =
    Text(text.uppercase(), style = MaterialTheme.typography.labelSmall, color = Palette.Muted)

@Composable
private fun RowScope.Metric(label: String, value: String, color: Color = Palette.Text) =
    Column(
        Modifier.weight(1f).clip(RoundedCornerShape(12.dp)).background(Palette.Surface2).padding(10.dp),
    ) {
        Text(label.uppercase(), style = MaterialTheme.typography.labelSmall, color = Palette.Muted, fontSize = 10.sp)
        Text(value, color = color, fontFamily = Mono, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, maxLines = 1)
    }

@Composable
private fun OfflineCard(info: String) = Card {
    Text("Not connected", style = MaterialTheme.typography.titleMedium)
    Text(
        "Join the robot's Wi-Fi and tap Connect. The phone becomes the console: it sends the " +
            "heartbeat, and closing the app e-stops the robot.",
        color = Palette.Muted, fontSize = 14.sp,
    )
    if (info.isNotEmpty() && info != "stopped") Text(info, color = Palette.Warn, fontFamily = Mono, fontSize = 12.sp)
}

private fun Double.f(n: Int) = "%.${n}f".format(this)

// ---------------------------------------------------------------- drive

@Composable
private fun DriveTab(s: Status, core: ConsoleCore) {
    val rs = s.robotState
    val vel = rs?.optJSONArray("velocity")
    Card {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Metric("Mode", s.mode.uppercase(), if (s.mode == "policy") Palette.Accent else Palette.Text)
            Metric("Posture", rs?.optString("posture") ?: "—")
            Metric("Tilt", rs?.let { "${it.optDouble("tilt_deg").f(1)}°" } ?: "—")
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Metric("vx", vel?.optDouble(0)?.f(2) ?: "—")
            Metric("vy", vel?.optDouble(1)?.f(2) ?: "—")
            Metric("wz", vel?.optDouble(2)?.f(2) ?: "—")
        }
    }

    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Phone drive", style = MaterialTheme.typography.titleMedium)
                Text(
                    if (s.driveEnabled) "Overriding Nav2 (${ConsoleCore.CMD_VEL})" else "Off — Nav2 and others drive",
                    color = if (s.driveEnabled) Palette.Accent else Palette.Muted, fontSize = 13.sp,
                )
            }
            Switch(
                s.driveEnabled, { core.setDrive(it) }, enabled = !s.estopLatched,
                colors = SwitchDefaults.colors(checkedTrackColor = Palette.Accent, checkedThumbColor = Palette.Bg),
            )
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Turbo", Modifier.weight(1f), color = if (s.turbo) Palette.Warn else Palette.Muted)
            Text(
                if (s.turbo) "1.0 m/s · 1.0 rad/s" else "0.5 m/s · 0.5 rad/s",
                color = Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
            )
            Spacer(Modifier.width(10.dp))
            Switch(
                s.turbo, { core.setTurbo(it) },
                colors = SwitchDefaults.colors(checkedTrackColor = Palette.Warn, checkedThumbColor = Palette.Bg),
            )
        }
    }

    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Joystick(s.driveEnabled, horizontalOnly = false, onMove = { x, y -> core.setMove(-y.toDouble(), -x.toDouble()) })
            Spacer(Modifier.height(6.dp))
            SectionLabel("Move")
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Joystick(s.driveEnabled, horizontalOnly = true, onMove = { x, _ -> core.setTurn(-x.toDouble()) })
            Spacer(Modifier.height(6.dp))
            SectionLabel("Turn")
        }
    }
    val c = s.driveCmd
    Text(
        "cmd  vx ${c[0].f(2)}  vy ${c[1].f(2)}  wz ${c[2].f(2)}",
        Modifier.fillMaxWidth(), textAlign = TextAlign.Center,
        color = if (s.driveEnabled) Palette.Text else Palette.Muted, fontFamily = Mono, fontSize = 13.sp,
    )
    if (s.driveEnabled && s.mode != "policy") {
        Text(
            "Robot is in POSE mode — switch to Policy on the Robot tab to walk.",
            Modifier.fillMaxWidth(), textAlign = TextAlign.Center, color = Palette.Warn, fontSize = 12.sp,
        )
    }
}

// ---------------------------------------------------------------- robot

@Composable
private fun Segmented(options: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Palette.Surface2).padding(4.dp),
    ) {
        options.forEachIndexed { i, o ->
            val bg by animateColorAsState(if (i == selected) Palette.Accent else Color.Transparent, label = "seg")
            Box(
                Modifier.weight(1f).clip(RoundedCornerShape(10.dp)).background(bg).clickable { onSelect(i) }
                    .padding(vertical = 12.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(o, color = if (i == selected) Palette.Bg else Palette.Text, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

@Composable
private fun RobotTab(s: Status, core: ConsoleCore) {
    var refusal by remember { mutableStateOf<List<String>>(emptyList()) }
    Card {
        SectionLabel("Control mode")
        Segmented(listOf("Pose", "Policy"), if (s.mode == "policy") 1 else 0) {
            refusal = core.setMode(if (it == 1) "policy" else "pose")
        }
        if (refusal.isNotEmpty()) {
            Column(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Palette.Danger.copy(alpha = 0.12f))
                    .padding(12.dp),
            ) {
                Text("Policy refused", color = Palette.Danger, fontWeight = FontWeight.Bold)
                refusal.forEach { Text("• $it", color = Palette.Text, fontSize = 13.sp) }
                Spacer(Modifier.height(6.dp))
                OutlinedButton(
                    { refusal = core.setMode("policy", force = true) },
                    border = BorderStroke(1.dp, Palette.Danger),
                ) { Text("Force policy anyway", color = Palette.Danger) }
            }
        }
    }

    if (s.mode == "policy") {
        Card {
            Row(verticalAlignment = Alignment.CenterVertically) {
                SectionLabel("Live robot")
                Spacer(Modifier.weight(1f))
                Text("drag to orbit · pinch to zoom", color = Palette.Muted, fontSize = 11.sp)
            }
            Robot3D(Modifier.fillMaxWidth().height(320.dp).clip(RoundedCornerShape(14.dp)).background(Palette.Bg))
            val rs = s.robotState
            val vel = rs?.optJSONArray("velocity")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Metric("Posture", rs?.optString("posture") ?: "—")
                Metric("vx", vel?.optDouble(0)?.f(2) ?: "—")
                Metric("wz", vel?.optDouble(2)?.f(2) ?: "—")
            }
        }
    } else {
        Card {
            Row(verticalAlignment = Alignment.CenterVertically) {
                SectionLabel("Poses")
                Spacer(Modifier.weight(1f))
                Text(
                    "${s.poseName}${if (s.poseLabel.isNotEmpty()) " · ${s.poseLabel}" else ""}",
                    color = Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
                )
            }
            LinearProgressIndicator(
                { s.poseProgress.toFloat() }, Modifier.fillMaxWidth().height(6.dp).clip(CircleShape),
                color = Palette.Accent, trackColor = Palette.Surface2,
            )
            val poses = listOf("stand" to "Stand", "sit" to "Sit", "lie_flat" to "Lie flat", "pushup" to "Push-up")
            poses.chunked(2).forEach { row ->
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    row.forEach { (key, label) ->
                        val active = s.poseName == key
                        Box(
                            Modifier.weight(1f).height(64.dp).clip(RoundedCornerShape(14.dp))
                                .background(if (active) Palette.Accent.copy(alpha = 0.18f) else Palette.Surface2)
                                .border(1.dp, if (active) Palette.Accent else Color.Transparent, RoundedCornerShape(14.dp))
                                .clickable { core.sendPose(key) },
                            contentAlignment = Alignment.Center,
                        ) { Text(label, fontWeight = FontWeight.SemiBold, color = if (active) Palette.Accent else Palette.Text) }
                    }
                }
            }
            var interp by remember { mutableStateOf(3f) }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Transition", color = Palette.Muted, fontSize = 13.sp)
                Slider(
                    interp, { interp = it }, Modifier.weight(1f).padding(horizontal = 10.dp), valueRange = 0.5f..6f,
                    onValueChangeFinished = { core.setInterp((Math.round(interp * 10) / 10.0)) },
                    colors = sliderColors(),
                )
                Text("${"%.1f".format(interp)} s", fontFamily = Mono, fontSize = 13.sp)
            }
        }
    }

    Card {
        OutlinedButton(
            { core.safetyReset() }, Modifier.fillMaxWidth(), border = BorderStroke(1.dp, Palette.Warn),
        ) { Text("Safety reset", color = Palette.Warn) }
        val rs = s.robotState
        if (rs != null) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Metric("Torque", if (rs.optBoolean("torque_on")) "ON" else "OFF", if (rs.optBoolean("torque_on")) Palette.Ok else Palette.Muted)
                Metric("Safety", if (rs.optBoolean("safety_blocked")) "BLOCKED" else "OK", if (rs.optBoolean("safety_blocked")) Palette.Danger else Palette.Ok)
            }
        }
    }

    SystemCard(s)
}

private fun loadColor(pct: Double) = when {
    pct >= 85 -> Palette.Danger
    pct >= 60 -> Palette.Warn
    else -> Palette.Accent
}

/** /system_stats from the robot's system_monitor: CPU per core, GPU, RAM, temperatures. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun SystemCard(s: Status) {
    val st = s.systemStats
    val stale = s.systemStatsAgeMs !in 0..3000
    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SectionLabel("Robot computer")
            Spacer(Modifier.weight(1f))
            Text(
                when {
                    st == null -> "no /system_stats"
                    stale -> "stale ${s.systemStatsAgeMs / 1000}s"
                    else -> "load ${st.optJSONArray("load")?.optDouble(0)?.f(2) ?: "—"}"
                },
                color = if (st == null || stale) Palette.Warn else Palette.Muted, fontSize = 12.sp, fontFamily = Mono,
            )
        }
        if (st == null) {
            Text("Start the real driver with monitor:=true (the default).", color = Palette.Muted, fontSize = 13.sp)
            return@Card
        }
        val cpu = st.optDouble("cpu", 0.0)
        val gpu = if (st.isNull("gpu")) null else st.optDouble("gpu")
        val temp = if (st.isNull("temp_max")) null else st.optDouble("temp_max")
        val used = st.optDouble("ram_used_mb", 0.0) / 1024
        val total = st.optDouble("ram_total_mb", 0.0) / 1024
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Metric("CPU", "${cpu.toInt()}%", loadColor(cpu))
            Metric("GPU", gpu?.let { "${it.toInt()}%" } ?: "—", gpu?.let(::loadColor) ?: Palette.Muted)
            Metric("Temp", temp?.let { "${it.toInt()}°C" } ?: "—", when {
                temp == null -> Palette.Muted
                temp >= 80 -> Palette.Danger
                temp >= 65 -> Palette.Warn
                else -> Palette.Text
            })
        }
        Column {
            Row {
                Text("RAM", Modifier.weight(1f), color = Palette.Muted, fontSize = 12.sp)
                Text("${used.f(1)} / ${total.f(1)} GB", fontFamily = Mono, fontSize = 12.sp)
            }
            Spacer(Modifier.height(4.dp))
            val frac = if (total > 0) used / total else 0.0
            LinearProgressIndicator(
                { frac.toFloat() }, Modifier.fillMaxWidth().height(6.dp).clip(CircleShape),
                color = loadColor(frac * 100), trackColor = Palette.Surface2,
            )
        }
        val cores = st.optJSONArray("cores")
        if (cores != null && cores.length() > 0) {
            Text("CORES", style = MaterialTheme.typography.labelSmall, color = Palette.Muted, fontSize = 10.sp)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                for (i in 0 until cores.length()) {
                    val pct = cores.optDouble(i, 0.0)
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Box(
                            Modifier.width(30.dp).height(44.dp).clip(RoundedCornerShape(6.dp)).background(Palette.Surface2),
                            contentAlignment = Alignment.BottomCenter,
                        ) {
                            Box(
                                Modifier.fillMaxWidth().height((44 * pct / 100).coerceIn(0.0, 44.0).dp)
                                    .background(loadColor(pct)),
                            )
                        }
                        Text("$i", color = Palette.Muted, fontSize = 10.sp, fontFamily = Mono)
                    }
                }
            }
        }
        val temps = st.optJSONObject("temps")
        if (temps != null && temps.length() > 0) {
            Text(
                temps.keys().asSequence().sorted().joinToString("  ") { "$it ${temps.optDouble(it).toInt()}°" },
                color = Palette.Muted, fontSize = 11.sp, fontFamily = Mono,
            )
        }
    }
}

@Composable
private fun sliderColors() = SliderDefaults.colors(
    thumbColor = Palette.Accent, activeTrackColor = Palette.Accent, inactiveTrackColor = Palette.Surface2,
    activeTickColor = Color.Transparent, inactiveTickColor = Color.Transparent,
)

// ---------------------------------------------------------------- tune

private class ParamSpec(
    val label: String, val range: ClosedFloatingPointRange<Float>, val step: Float,
    val get: (SafetyParams) -> Double, val set: (SafetyParams, Double) -> SafetyParams, val fmt: (Double) -> String,
)

private val PARAMS = listOf(
    ParamSpec("Torque limit", 0f..100f, 1f, { it.torquePercent }, { p, v -> p.copy(torquePercent = v) }, { "${it.toInt()} % · ${(it / 100 * 45).f(1)} Nm" }),
    ParamSpec("Max roll", 5f..60f, 1f, { it.rollLimitDeg }, { p, v -> p.copy(rollLimitDeg = v) }, { "${it.toInt()}°" }),
    ParamSpec("Max pitch", 5f..60f, 1f, { it.pitchLimitDeg }, { p, v -> p.copy(pitchLimitDeg = v) }, { "${it.toInt()}°" }),
    ParamSpec("Joint ROM margin", 0f..0.5f, 0.01f, { it.romMargin }, { p, v -> p.copy(romMargin = v) }, { "${(it * 100).toInt()} %" }),
    ParamSpec("Watchdog", 0.1f..2f, 0.05f, { it.watchdogTimeout }, { p, v -> p.copy(watchdogTimeout = v) }, { "${it.f(2)} s" }),
    ParamSpec("Kp", 0f..80f, 0.5f, { it.kp }, { p, v -> p.copy(kp = v) }, { it.f(1) }),
    ParamSpec("Kd", 0f..5f, 0.05f, { it.kd }, { p, v -> p.copy(kd = v) }, { it.f(2) }),
)

@Composable
private fun TuneTab(s: Status?, core: ConsoleCore?) {
    val ctx = LocalContext.current
    var saved by remember { mutableStateOf(Settings.loadParams(ctx)) }
    val applied = s?.params ?: saved
    var draft by remember(applied) { mutableStateOf(applied) }
    var error by remember { mutableStateOf("") }
    val dirty = draft != applied

    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SectionLabel("Safety & gains")
            Spacer(Modifier.weight(1f))
            if (core == null) Text("saved for next connect", color = Palette.Muted, fontSize = 12.sp)
        }
        PARAMS.forEach { spec ->
            val v = spec.get(draft)
            val changed = v != spec.get(applied)
            Column {
                Row {
                    Text(spec.label, Modifier.weight(1f), fontSize = 14.sp)
                    Text(spec.fmt(v), color = if (changed) Palette.Warn else Palette.Muted, fontFamily = Mono, fontSize = 13.sp)
                }
                Slider(
                    v.toFloat(),
                    { draft = spec.set(draft, (Math.round(it / spec.step) * spec.step).toDouble().let { x -> Math.round(x * 1000) / 1000.0 }) },
                    valueRange = spec.range, colors = sliderColors(),
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(
                { draft = applied }, Modifier.weight(1f), enabled = dirty, border = BorderStroke(1.dp, Palette.Outline),
            ) { Text("Revert", color = Palette.Muted) }
            Button(
                {
                    error = try {
                        core?.updateParams(draft) ?: draft.validated()
                        Settings.saveParams(ctx, draft)
                        saved = draft
                        ""
                    } catch (e: Exception) {
                        e.message ?: "invalid"
                    }
                },
                Modifier.weight(1f), enabled = dirty,
                colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.Bg),
            ) { Text("Apply", fontWeight = FontWeight.Bold) }
        }
        if (error.isNotEmpty()) Text(error, color = Palette.Danger, fontSize = 13.sp)
        if (s?.estopLatched == true) Text("E-stop latched: torque is held at 0 % regardless.", color = Palette.Danger, fontSize = 12.sp)
    }
}

// ---------------------------------------------------------------- link

@Composable
private fun LinkTab(s: Status?, info: String) {
    val ctx = LocalContext.current
    Card {
        SectionLabel("DDS link")
        Text(info, fontFamily = Mono, fontSize = 12.sp, color = Palette.Text)
        if (s != null) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Metric("Heartbeat", "#${s.heartbeatCount}")
                Metric("Readers", "${s.heartbeatReaders}", if (s.heartbeatReaders > 0) Palette.Ok else Palette.Danger)
                Metric("State age", if (s.robotStateAgeMs < 0) "—" else "${s.robotStateAgeMs} ms")
            }
            if (s.heartbeatReaders == 0) {
                Text("No robot is reading the heartbeat yet.", color = Palette.Warn, fontSize = 12.sp)
            }
        }
    }

    val cur = remember { Settings.loadLink(ctx) }
    var domain by remember { mutableStateOf(cur.domainId.toString()) }
    var mode by remember { mutableStateOf(cur.discoveryMode) }
    var server by remember { mutableStateOf(cur.discoveryServer) }
    var hz by remember { mutableStateOf(cur.heartbeatHz.toString()) }
    var saved by remember { mutableStateOf(false) }
    val editable = s == null
    Card {
        Row {
            SectionLabel("Settings")
            Spacer(Modifier.weight(1f))
            if (!editable) Text("disconnect to edit", color = Palette.Muted, fontSize = 12.sp)
        }
        OutlinedTextField(
            domain, { domain = it; saved = false }, Modifier.fillMaxWidth(), enabled = editable,
            label = { Text("ROS_DOMAIN_ID") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        )
        Text("Discovery server", color = Palette.Muted, fontSize = 13.sp)
        val modes = listOf("auto", "on", "off")
        Segmented(listOf("Auto", "Always", "Multicast"), modes.indexOf(mode).coerceAtLeast(0)) {
            if (editable) {
                mode = modes[it]
                saved = false
            }
        }
        OutlinedTextField(
            server, { server = it; saved = false }, Modifier.fillMaxWidth(), enabled = editable,
            label = { Text("Server ip:port") }, singleLine = true,
        )
        OutlinedTextField(
            hz, { hz = it; saved = false }, Modifier.fillMaxWidth(), enabled = editable,
            label = { Text("Heartbeat Hz") }, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        )
        Button(
            {
                Settings.saveLink(
                    ctx,
                    LinkSettings(domain.toIntOrNull() ?: 42, mode, server.trim(), hz.toDoubleOrNull()?.takeIf { it > 0 } ?: 10.0),
                )
                saved = true
            },
            Modifier.fillMaxWidth(), enabled = editable,
            colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.Bg),
        ) { Text(if (saved) "Saved" else "Save", fontWeight = FontWeight.Bold) }
    }

    if (s != null && s.events.isNotEmpty()) {
        Card {
            SectionLabel("Events")
            s.events.forEach { Text(it, fontSize = 12.sp, color = Palette.Muted) }
        }
    }
}
