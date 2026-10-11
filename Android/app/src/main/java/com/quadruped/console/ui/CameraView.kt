package com.quadruped.console.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quadruped.console.MsgType
import com.quadruped.console.NativeLink
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/** camera_bridge's list of cameras and their preview settings (JSON, latched). */
const val CAMERAS = "/cameras"

private val RATES = listOf(1.0, 2.0, 5.0, 10.0)
private val WIDTHS = listOf(160, 320, 640)

private data class CameraEntry(
    val name: String, val kind: String, val preview: String,
    val rate: Double, val previewRate: Double, val previewWidth: Int,
) {
    val settings get() = preview.removeSuffix("/compressed") + "/settings"
}

private fun parseCameras(json: String): List<CameraEntry> = runCatching {
    val a = JSONArray(json)
    (0 until a.length()).map {
        val o = a.getJSONObject(it)
        CameraEntry(
            o.getString("name"), o.optString("kind", "rgb"), o.getString("preview"),
            o.optDouble("rate", 15.0), o.optDouble("preview_rate", 2.0), o.optInt("preview_width", 320),
        )
    }
}.getOrDefault(emptyList())

/**
 * The robot's cameras, as camera_bridge's small, slow previews (best effort, so a lost frame is just skipped).
 * Only the chosen camera is subscribed, and nothing once the tab is left: the robot then stops making previews.
 * Rate and width go to the robot, which echoes them back in /cameras; until it does, the request is resent.
 */
@Composable
fun CameraView(heartbeat: JSONObject?, watchdogS: Double, modifier: Modifier = Modifier) {
    var cameras by remember { mutableStateOf(emptyList<CameraEntry>()) }
    var selected by remember { mutableStateOf<String?>(null) }
    var frame by remember { mutableStateOf<ImageBitmap?>(null) }
    var frameBytes by remember { mutableIntStateOf(0) }
    var ageMs by remember { mutableLongStateOf(-1L) }
    var request by remember { mutableStateOf<Pair<Double, Int>?>(null) }
    val cam = cameras.firstOrNull { it.name == selected }
    val topic = cam?.preview

    DisposableEffect(Unit) { onDispose { NativeLink.unsubscribe(CAMERAS) } }
    DisposableEffect(topic) {
        onDispose {
            topic?.let { NativeLink.unsubscribe(it) }
            frame = null
            frameBytes = 0
            ageMs = -1L
            request = null
        }
    }
    LaunchedEffect(topic) {
        var count = 0L
        var listText: String? = null
        var sentMs = 0L
        var sent: Pair<Double, Int>? = null
        var tries = 0
        cam?.let { NativeLink.advertise(it.settings, MsgType.STRING) }
        withContext(Dispatchers.Default) {
            while (true) {
                NativeLink.subscribe(CAMERAS, MsgType.STRING, reliable = true, transientLocal = true)
                NativeLink.latestText(CAMERAS)?.takeIf { it != listText }?.let {
                    listText = it
                    cameras = parseCameras(it)
                    if (cameras.none { c -> c.name == selected }) selected = cameras.firstOrNull()?.name
                }
                val c = cameras.firstOrNull { it.name == selected }
                if (topic != null && c != null) {
                    NativeLink.subscribe(topic, MsgType.COMPRESSED_IMAGE, reliable = false)
                    val n = NativeLink.latestCount(topic)
                    if (n != count) {
                        count = n
                        NativeLink.latestBytes(topic)?.let { b ->
                            frameBytes = b.size
                            BitmapFactory.decodeByteArray(b, 0, b.size)?.let { frame = it.asImageBitmap() }
                        }
                    }
                    ageMs = NativeLink.latestAgeMs(topic)
                    val r = request
                    val now = System.currentTimeMillis()
                    if (r != sent) tries = 0
                    sent = r
                    if (r != null && r != (c.previewRate to c.previewWidth) && now - sentMs > 1000) {
                        // The robot clamps what it can't do; give up rather than resend forever.
                        if (tries++ >= 5) request = null
                        else {
                            sentMs = now
                            NativeLink.publishString(c.settings, """{"rate": ${r.first}, "width": ${r.second}}""")
                        }
                    }
                }
                delay(100)
            }
        }
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (cameras.size > 1) Choice(cameras.map { it.name }, cameras.indexOfFirst { it.name == selected }) {
            selected = cameras[it].name
        }
        Box(
            Modifier.fillMaxWidth().weight(1f).clip(RoundedCornerShape(18.dp)).background(Palette.Surface),
            contentAlignment = Alignment.Center,
        ) {
            val f = frame
            if (f != null) {
                Image(f, selected, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            } else {
                Text(
                    if (cameras.isEmpty()) "no cameras (start them with the launcher's [I])" else "waiting for $selected",
                    color = Palette.Muted,
                )
            }
            if (f != null) Text(
                "$selected · ${if (ageMs in 0..1500) "live" else "${ageMs / 1000} s old"}",
                color = if (ageMs in 0..1500) Palette.Text else Palette.Warn, fontFamily = Mono,
                modifier = Modifier.align(Alignment.TopStart).padding(10.dp)
                    .clip(RoundedCornerShape(8.dp)).background(Palette.Bg.copy(alpha = 0.6f)).padding(6.dp, 3.dp),
            )
        }
        if (cam != null) {
            val want = request ?: (cam.previewRate to cam.previewWidth)
            val rates = RATES.filter { it <= cam.rate }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Rate", color = Palette.Muted, fontSize = 13.sp, modifier = Modifier.width(52.dp))
                Choice(rates.map { "${it.toInt()} Hz" }, rates.indexOf(want.first)) { request = rates[it] to want.second }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Width", color = Palette.Muted, fontSize = 13.sp, modifier = Modifier.width(52.dp))
                Choice(WIDTHS.map { "$it px" }, WIDTHS.indexOf(want.second)) { request = want.first to WIDTHS[it] }
            }
            val pending = request != null && request != (cam.previewRate to cam.previewWidth)
            Text(
                if (pending) "sending to the robot…"
                else "${cam.previewWidth} px · ${"%g".format(cam.previewRate)} Hz · " +
                    if (frameBytes > 0) "${frameBytes / 1024} kB a frame ≈ ${(frameBytes * cam.previewRate / 1024).toInt()} kB/s"
                    else "—",
                color = if (pending) Palette.Warn else Palette.Muted, fontFamily = Mono, fontSize = 12.sp,
            )
        }
        HeartbeatLine(heartbeat, watchdogS)
    }
}

/** The robot's view of our heartbeat: rate, heartbeats lost in the last 10 s and the longest gap. */
@Composable
fun HeartbeatLine(heartbeat: JSONObject?, watchdogS: Double) {
    if (heartbeat == null) {
        Text("Heartbeat: no report from the robot", color = Palette.Muted, fontFamily = Mono, fontSize = 12.sp)
        return
    }
    val missed = heartbeat.optInt("missed_10s")
    val gap = heartbeat.optInt("max_gap_ms")
    val color = when {
        gap >= watchdogS * 1000 -> Palette.Danger
        missed > 0 -> Palette.Warn
        else -> Palette.Ok
    }
    Text(
        "Heartbeat ${heartbeat.optDouble("hz").let { "%.1f".format(it) }} Hz · $missed lost in 10 s " +
            "(${heartbeat.optInt("missed")} total) · max gap $gap ms",
        color = color, fontFamily = Mono, fontSize = 12.sp,
    )
}

@Composable
private fun Choice(options: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Palette.Surface2).padding(3.dp)) {
        options.forEachIndexed { i, o ->
            val on = i == selected
            Box(
                Modifier.weight(1f).clip(RoundedCornerShape(9.dp)).background(if (on) Palette.Accent else Palette.Surface2)
                    .clickable { onSelect(i) }.padding(vertical = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(o, color = if (on) Palette.Bg else Palette.Text, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
            }
        }
    }
}
