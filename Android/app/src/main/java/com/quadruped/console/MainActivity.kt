package com.quadruped.console

import android.Manifest
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.displayCutoutPadding
import androidx.compose.foundation.layout.imePadding
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
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
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
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quadruped.console.ui.CameraView
import com.quadruped.console.ui.ConsoleTheme
import com.quadruped.console.ui.HeartbeatLine
import com.quadruped.console.ui.Joystick
import com.quadruped.console.ui.MapView
import com.quadruped.console.ui.Robot3D
import com.quadruped.console.ui.ShellTab
import com.quadruped.console.ui.Mono
import com.quadruped.console.ui.Palette
import kotlinx.coroutines.delay
import org.json.JSONObject

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= 33) requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 0)
        setContent { ConsoleTheme { ConsoleApp() } }
        hideStatusBar()
    }

    /** The status bar only takes room; a swipe from the top still shows it for a moment. */
    private fun hideStatusBar() {
        WindowCompat.getInsetsController(window, window.decorView).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.statusBars())
        }
    }

    // Some devices bring it back after the keyboard or a dialog.
    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) hideStatusBar()
    }

    /** Nobody is holding the sticks once the screen is gone. */
    override fun onPause() {
        ConsoleService.core?.setDrive(false)
        super.onPause()
    }
}

private enum class Tab(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    DRIVE("Drive", Icons.Filled.PlayArrow),
    MAP("Map", Icons.Filled.LocationOn),
    CAMERA("Camera", Icons.Filled.Search),
    SHELL("Shell", Icons.AutoMirrored.Filled.KeyboardArrowRight),
    SYSTEM("System", Icons.Filled.Settings),
}

@Composable
fun ConsoleApp() {
    var status by remember { mutableStateOf<Status?>(null) }
    var info by remember { mutableStateOf("") }
    var tab by rememberSaveable { mutableIntStateOf(0) }
    var map3d by rememberSaveable { mutableStateOf(false) }
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
            Modifier.fillMaxSize().padding(bottom = pad.calculateBottomPadding()).consumeWindowInsets(pad)
                .imePadding().statusBarsPadding().displayCutoutPadding().padding(horizontal = 16.dp),
        ) {
            val s = status
            val c = core
            TopBar(s, c != null, info)
            Spacer(Modifier.height(8.dp))
            EstopButton(s, c)
            Spacer(Modifier.height(8.dp))
            val body = Modifier.fillMaxWidth().weight(1f).padding(bottom = 8.dp)
            when (Tab.entries[tab]) {
                Tab.DRIVE -> DriveTab(s, c, body)
                Tab.MAP -> MapTab(map3d, { map3d = it }, body)
                Tab.CAMERA -> CameraView(
                    s?.robotState?.optJSONObject("heartbeat"), (s?.params ?: SafetyParams()).watchdogTimeout, body,
                )
                Tab.SHELL -> Column(body) { ShellTab(Modifier.weight(1f)) }
                Tab.SYSTEM -> Column(body.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    SystemTab(s, c, info)
                }
            }
        }
    }
}

// ---------------------------------------------------------------- chrome

@Composable
private fun TopBar(s: Status?, connected: Boolean, info: String) {
    val ctx = LocalContext.current
    Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        LinkPill(s, connected)
        Spacer(Modifier.width(8.dp))
        Box(Modifier.weight(1f)) {
            val rs = s?.robotState
            when {
                rs != null -> RobotSummary(rs)
                !connected && info.isNotEmpty() && info != "stopped" ->
                    Text(info, color = Palette.Warn, fontFamily = Mono, fontSize = 11.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
            }
        }
        Spacer(Modifier.width(8.dp))
        if (connected) {
            OutlinedButton(
                { ConsoleService.stop(ctx) }, Modifier.height(36.dp), border = BorderStroke(1.dp, Palette.Outline),
                contentPadding = PaddingValues(horizontal = 10.dp),
            ) { Text("Disconnect", color = Palette.Muted, fontSize = 13.sp) }
        } else {
            Button(
                { ConsoleService.start(ctx) }, Modifier.height(36.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.Bg),
                contentPadding = PaddingValues(horizontal = 14.dp),
            ) { Text("Connect", fontWeight = FontWeight.Bold) }
        }
    }
}

/** Posture and tilt, and the one thing most wrong with the robot, if anything. */
@Composable
private fun RobotSummary(rs: JSONObject) {
    val posture = rs.optString("posture").ifEmpty { "—" }
    val flag = when {
        rs.optBoolean("safety_blocked") -> "blocked" to Palette.Danger
        rs.optBoolean("numb") -> "numb" to Palette.Warn
        !rs.optBoolean("torque_on") -> "no torque" to Palette.Muted
        else -> null
    }
    Column {
        Text(
            posture, color = if (posture == "fallen") Palette.Danger else Palette.Text,
            fontWeight = FontWeight.SemiBold, fontSize = 14.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
        )
        Row {
            Text("${rs.optDouble("tilt_deg").f(0)}°", color = Palette.Muted, fontFamily = Mono, fontSize = 11.sp)
            flag?.let { (text, color) ->
                Text("  $text", color = color, fontSize = 11.sp, fontWeight = FontWeight.SemiBold, maxLines = 1)
            }
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
        val pct = s?.heartbeatLossPercent
        if (connected && s != null && pct != null) {
            val lossColor = when {
                pct >= 5.0 -> Palette.Danger
                pct > 0.0 -> Palette.Warn
                else -> Palette.Ok
            }
            Spacer(Modifier.width(8.dp))
            LossGraph(s.heartbeatLoss, lossColor, Modifier.size(width = 36.dp, height = 14.dp))
            Spacer(Modifier.width(4.dp))
            Text(
                if (pct == 0.0) "0%" else if (pct < 10) "%.1f%%".format(pct) else "%.0f%%".format(pct),
                color = lossColor, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono,
            )
        }
    }
}

/** Heartbeats the robot lost, one bar per second over the last minute; 5 % fills the height. */
@Composable
private fun LossGraph(loss: DoubleArray, color: Color, modifier: Modifier) = Canvas(modifier) {
    val n = 60
    val w = size.width / n
    val top = maxOf(0.05, loss.filter { !it.isNaN() }.maxOrNull() ?: 0.0)
    drawLine(color.copy(alpha = 0.3f), Offset(0f, size.height), Offset(size.width, size.height), 1f)
    loss.takeLast(n).forEachIndexed { i, v ->
        if (v.isNaN() || v <= 0.0) return@forEachIndexed
        val h = (v / top * size.height).toFloat().coerceAtLeast(2f)
        val x = (n - loss.size.coerceAtMost(n) + i) * w
        drawRect(color, Offset(x, size.height - h), Size(maxOf(w - 0.5f, 1f), h))
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
        Modifier.fillMaxWidth().height(if (latched) 64.dp else 52.dp).clip(shape)
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
                fontSize = if (latched) 18.sp else 24.sp,
                fontWeight = FontWeight.Black, letterSpacing = 4.sp,
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

private fun Double.f(n: Int) = "%.${n}f".format(this)

// ---------------------------------------------------------------- drive

/**
 * Mode, then only what that mode uses: poses in Pose; policies, tricks and the sticks in Policy, the
 * sticks docked at the bottom. Offline it is the same page, disabled.
 */
@Composable
private fun DriveTab(s: Status?, core: ConsoleCore?, modifier: Modifier) {
    val live = s != null && core != null
    val policy = s?.mode == "policy"
    // Sticks only walk in policy; leaving it hands control back to Nav2 and the others.
    LaunchedEffect(policy) { if (!policy && s?.driveEnabled == true) core?.setDrive(false) }
    var refusal by remember { mutableStateOf<List<String>>(emptyList()) }
    val rs = s?.robotState
    val numb = rs?.optBoolean("numb") == true

    Column(modifier, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Column(
            Modifier.fillMaxWidth().weight(1f).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            when {
                rs?.optBoolean("safety_policy_active") == true ->
                    Banner("Safety lie-down", "Tripped: ${rs.optString("trip_reason")}. Goes numb once lying still.", Palette.Danger)
                numb -> Banner("Numb", "${rs?.optString("numb_reason")}. Release to hold where it is in pose mode.", Palette.Warn)
            }
            if (rs?.optBoolean("safety_blocked") == true) {
                Banner("Safety blocked", "A safety limit tripped. Reset once the cause is gone.", Palette.Danger) {
                    OutlinedButton({ core?.safetyReset() }, border = BorderStroke(1.dp, Palette.Danger)) {
                        Text("Safety reset", color = Palette.Danger)
                    }
                }
            }
            if (refusal.isNotEmpty() && core != null) {
                Banner("Policy refused", refusal.joinToString("\n") { "• $it" }, Palette.Danger) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton({ refusal = core.setMode("policy", force = true) }, border = BorderStroke(1.dp, Palette.Danger)) {
                            Text("Force policy", color = Palette.Danger)
                        }
                        OutlinedButton({ refusal = emptyList() }, border = BorderStroke(1.dp, Palette.Outline)) {
                            Text("Dismiss", color = Palette.Muted)
                        }
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.weight(1f)) {
                    Segmented(listOf("Pose", "Policy"), if (policy) 1 else 0, enabled = live) {
                        refusal = core?.setMode(if (it == 1) "policy" else "pose") ?: emptyList()
                    }
                }
                OutlinedButton(
                    { core?.setNumb(!numb) }, Modifier.height(48.dp), enabled = live,
                    border = BorderStroke(1.dp, if (live) Palette.Warn else Palette.Outline), shape = RoundedCornerShape(14.dp),
                    contentPadding = PaddingValues(horizontal = 14.dp),
                ) { Text(if (numb) "Release" else "Numb", color = if (live) Palette.Warn else Palette.Muted) }
            }
            if (policy && s != null && core != null) {
                PolicyCard(rs, core)
                TrickCard(s, core)
            } else {
                PoseCard(s, core)
            }
        }
        if (policy && s != null && core != null) DriveDock(s, core)
    }
}

@Composable
private fun Banner(title: String, body: String, color: Color, action: (@Composable () -> Unit)? = null) = Column(
    Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(color.copy(alpha = 0.12f)).padding(12.dp),
) {
    Text(title, color = color, fontWeight = FontWeight.Bold)
    Text(body, color = Palette.Text, fontSize = 13.sp)
    if (action != null) {
        Spacer(Modifier.height(6.dp))
        action()
    }
}

private fun JSONObject.stringOrNull(key: String) = if (isNull(key)) null else optString(key)

/** The policies /robot_state lists; the robot loads the one tapped and switches once it is ready. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PolicyCard(rs: JSONObject?, core: ConsoleCore) = Card {
    val names = rs?.optJSONArray("policies")?.let { a -> List(a.length()) { a.optString(it) } }.orEmpty()
    val active = rs?.stringOrNull("policy")
    val loading = rs?.stringOrNull("policy_loading")
    val checks = rs?.optJSONArray("protections")?.let { a -> List(a.length()) { a.optString(it) } }.orEmpty()
    Row(verticalAlignment = Alignment.CenterVertically) {
        SectionLabel("Policy")
        Spacer(Modifier.weight(1f))
        Text(
            if (loading != null) "loading $loading…" else "checks: ${checks.joinToString(" ").ifEmpty { "none" }}",
            color = Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
        )
    }
    if (names.isEmpty()) {
        Text("The robot lists no policies.", color = Palette.Muted, fontSize = 13.sp)
        return@Card
    }
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        names.forEach { n ->
            val on = n == active
            val color = if (n == loading) Palette.Warn else Palette.Accent
            Box(
                Modifier.height(40.dp).clip(RoundedCornerShape(12.dp))
                    .background(if (on || n == loading) color.copy(alpha = 0.18f) else Palette.Surface2)
                    .border(1.dp, if (on || n == loading) color else Color.Transparent, RoundedCornerShape(12.dp))
                    .clickable { core.setPolicy(n) }.padding(horizontal = 14.dp),
                contentAlignment = Alignment.Center,
            ) { Text(n, fontWeight = FontWeight.SemiBold, color = if (on || n == loading) color else Palette.Text) }
        }
    }
    rs?.stringOrNull("policy_error")?.let { Text(it, color = Palette.Danger, fontSize = 12.sp) }
}

/**
 * For a trick policy (Pee, Sniff, Wiggle): hold a button to do the trick, let go and it stands back up. Pee has
 * two switches, one per leg.
 */
@Composable
private fun TrickCard(s: Status, core: ConsoleCore) {
    val rs = s.robotState ?: return
    val switches = rs.optInt("trick_switches", 0)
    if (switches <= 0) return
    val name = rs.optString("policy")
    val labels = if (switches == 2) listOf("Left", "Right") else listOf("Hold: $name")
    Card {
        Row(verticalAlignment = Alignment.CenterVertically) {
            SectionLabel("Trick")
            Spacer(Modifier.weight(1f))
            Text(
                if (s.trickHeld >= 0) "$name ${labels[s.trickHeld.coerceAtMost(labels.lastIndex)]}" else "hold to do it",
                color = if (s.trickHeld >= 0) Palette.Accent else Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            labels.forEachIndexed { i, label ->
                val held = s.trickHeld == i
                Box(
                    Modifier.weight(1f).height(64.dp).clip(RoundedCornerShape(14.dp))
                        .background(if (held) Palette.Accent.copy(alpha = 0.18f) else Palette.Surface2)
                        .border(1.dp, if (held) Palette.Accent else Color.Transparent, RoundedCornerShape(14.dp))
                        .pointerInput(i, s.estopLatched) {
                            if (s.estopLatched) return@pointerInput
                            detectTapGestures(onPress = {
                                core.setTrick(i)
                                try {
                                    awaitRelease()
                                } finally {
                                    core.setTrick(-1)
                                }
                            })
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        label, fontWeight = FontWeight.SemiBold,
                        color = if (s.estopLatched) Palette.Muted else if (held) Palette.Accent else Palette.Text,
                    )
                }
            }
        }
    }
}

/** The sticks, fixed at the bottom of the page so it scrolls without touching them. */
@Composable
private fun DriveDock(s: Status, core: ConsoleCore) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Palette.Surface)
            .border(1.dp, Palette.Outline.copy(alpha = 0.6f), RoundedCornerShape(18.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(
                s.driveEnabled, { core.setDrive(it) }, enabled = !s.estopLatched,
                colors = SwitchDefaults.colors(checkedTrackColor = Palette.Accent, checkedThumbColor = Palette.Bg),
            )
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("Phone drive", fontWeight = FontWeight.SemiBold)
                Text(
                    if (s.driveEnabled) "Overriding Nav2" else "Off: Nav2 and others drive",
                    color = if (s.driveEnabled) Palette.Accent else Palette.Muted, fontSize = 12.sp,
                )
            }
            Box(
                Modifier.clip(RoundedCornerShape(50)).background(if (s.turbo) Palette.Warn else Palette.Surface2)
                    .clickable { core.setTurbo(!s.turbo) }.padding(horizontal = 14.dp, vertical = 8.dp),
            ) {
                Text(
                    if (s.turbo) "Turbo 1.0" else "Turbo", color = if (s.turbo) Palette.Bg else Palette.Text,
                    fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                )
            }
        }
        BoxWithConstraints(Modifier.fillMaxWidth()) {
            val d = minOf(168.dp, (maxWidth - 16.dp) / 2)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Joystick(s.driveEnabled, horizontalOnly = false, diameter = d, onMove = { x, y -> core.setMove(-y.toDouble(), -x.toDouble()) })
                Joystick(s.driveEnabled, horizontalOnly = true, diameter = d, onMove = { x, _ -> core.setTurn(-x.toDouble()) })
            }
        }
        val c = s.driveCmd
        val vel = s.robotState?.optJSONArray("velocity")
        Text(
            "cmd ${c[0].f(2)} ${c[1].f(2)} ${c[2].f(2)}   meas ${vel?.optDouble(0)?.f(2) ?: "—"} ${vel?.optDouble(1)?.f(2) ?: "—"} ${vel?.optDouble(2)?.f(2) ?: "—"}",
            Modifier.fillMaxWidth(), textAlign = TextAlign.Center,
            color = if (s.driveEnabled) Palette.Text else Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
        )
    }
}

@Composable
private fun PoseCard(s: Status?, core: ConsoleCore?) = Card {
    val live = core != null
    Row(verticalAlignment = Alignment.CenterVertically) {
        SectionLabel("Poses")
        Spacer(Modifier.weight(1f))
        if (s != null) {
            Text(
                "${s.poseName}${if (s.poseLabel.isNotEmpty()) " · ${s.poseLabel}" else ""}",
                color = Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
            )
        }
    }
    LinearProgressIndicator(
        { (s?.poseProgress ?: 0.0).toFloat() }, Modifier.fillMaxWidth().height(6.dp).clip(CircleShape),
        color = Palette.Accent, trackColor = Palette.Surface2,
    )
    val poses = listOf("stand" to "Stand", "sit" to "Sit", "lie_flat" to "Lie flat", "pushup" to "Push-up")
    poses.chunked(2).forEach { row ->
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            row.forEach { (key, label) ->
                val active = s?.poseName == key
                Box(
                    Modifier.weight(1f).height(64.dp).clip(RoundedCornerShape(14.dp))
                        .background(if (active) Palette.Accent.copy(alpha = 0.18f) else Palette.Surface2)
                        .border(1.dp, if (active) Palette.Accent else Color.Transparent, RoundedCornerShape(14.dp))
                        .clickable(enabled = live) { core?.sendPose(key) },
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        label, fontWeight = FontWeight.SemiBold,
                        color = if (active) Palette.Accent else if (live) Palette.Text else Palette.Muted,
                    )
                }
            }
        }
    }
    var interp by remember { mutableStateOf(3f) }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("Transition", color = Palette.Muted, fontSize = 13.sp)
        Slider(
            interp, { interp = it }, Modifier.weight(1f).padding(horizontal = 10.dp), enabled = live, valueRange = 0.5f..6f,
            onValueChangeFinished = { core?.setInterp((Math.round(interp * 10) / 10.0)) },
            colors = sliderColors(),
        )
        Text("${"%.1f".format(interp)} s", fontFamily = Mono, fontSize = 13.sp)
    }
}

@Composable
private fun Segmented(options: List<String>, selected: Int, enabled: Boolean = true, onSelect: (Int) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Palette.Surface2).padding(4.dp),
    ) {
        options.forEachIndexed { i, o ->
            val on = i == selected && enabled
            val bg by animateColorAsState(if (on) Palette.Accent else Color.Transparent, label = "seg")
            Box(
                Modifier.weight(1f).clip(RoundedCornerShape(10.dp)).background(bg).clickable(enabled = enabled) { onSelect(i) }
                    .padding(vertical = 12.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(o, color = if (on) Palette.Bg else if (enabled) Palette.Text else Palette.Muted, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}

// ---------------------------------------------------------------- map

/** Nav2's 2D map or the 3D view in Point-LIO's voxels, one tab, switched from the corner. */
@Composable
private fun MapTab(threeD: Boolean, onToggle: (Boolean) -> Unit, modifier: Modifier) {
    Box(modifier.clip(RoundedCornerShape(18.dp)).background(Palette.Surface)) {
        if (threeD) Robot3D(Modifier.fillMaxSize()) else MapView(Modifier.fillMaxSize())
        Row(
            Modifier.align(Alignment.TopStart).padding(10.dp).clip(RoundedCornerShape(50))
                .background(Palette.Surface.copy(alpha = 0.9f)).padding(3.dp),
        ) {
            listOf("2D" to false, "3D" to true).forEach { (label, v) ->
                val on = v == threeD
                Box(
                    Modifier.clip(RoundedCornerShape(50)).background(if (on) Palette.Accent else Color.Transparent)
                        .clickable { onToggle(v) }.padding(horizontal = 14.dp, vertical = 5.dp),
                ) { Text(label, color = if (on) Palette.Bg else Palette.Text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold) }
            }
        }
    }
}

// ---------------------------------------------------------------- system

/** Health first (Jetson, link, events), then the settings, which rarely change. */
@Composable
private fun SystemTab(s: Status?, core: ConsoleCore?, info: String) {
    if (s != null) SystemCard(s)
    LinkCard(s, info)
    if (s != null && s.events.isNotEmpty()) {
        Card {
            SectionLabel("Events")
            s.events.forEach { Text(it, fontSize = 12.sp, color = Palette.Muted) }
        }
    }
    ParamsCard(s, core)
    ConnectionCard(s)
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
private fun ParamsCard(s: Status?, core: ConsoleCore?) {
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
        OutlinedButton(
            { core?.safetyReset() }, Modifier.fillMaxWidth(), enabled = core != null, border = BorderStroke(1.dp, Palette.Warn),
        ) { Text("Safety reset", color = Palette.Warn) }
        if (error.isNotEmpty()) Text(error, color = Palette.Danger, fontSize = 13.sp)
        if (s?.estopLatched == true) Text("E-stop latched: torque is held at 0 % regardless.", color = Palette.Danger, fontSize = 12.sp)
    }
}

@Composable
private fun LinkCard(s: Status?, info: String) =
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
            HeartbeatLine(s.robotState?.optJSONObject("heartbeat"), s.params.watchdogTimeout)
        }
    }

@Composable
private fun ConnectionCard(s: Status?) {
    val ctx = LocalContext.current
    val cur = remember { Settings.loadLink(ctx) }
    var domain by remember { mutableStateOf(cur.domainId.toString()) }
    var mode by remember { mutableStateOf(cur.discoveryMode) }
    var server by remember { mutableStateOf(cur.discoveryServer) }
    var hz by remember { mutableStateOf(cur.heartbeatHz.toString()) }
    var saved by remember { mutableStateOf(false) }
    val editable = s == null
    Card {
        Row {
            SectionLabel("Connection")
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
}
