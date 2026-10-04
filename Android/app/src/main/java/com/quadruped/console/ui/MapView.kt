package com.quadruped.console.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PointMode
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quadruped.console.MsgType
import com.quadruped.console.NativeLink
import com.quadruped.console.Pose3
import com.quadruped.console.TfTree
import com.quadruped.console.Vec3
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

private const val MAP = "/map"
private const val SCAN = "/scan"
private const val PLAN = "/plan"
private const val GOAL = "/goal_pose"

private class Grid(val image: ImageBitmap, val ox: Double, val oy: Double, val res: Double, val w: Int, val h: Int)

private class MapScene(
    val fixed: String,
    val robot: Pose3?,
    val scan: List<Vec3>,
    val plan: List<Vec3>,
)

private fun gridBitmap(v: DoubleArray, cells: ByteArray): Grid {
    val w = v[0].toInt()
    val h = v[1].toInt()
    val px = IntArray(w * h)
    val unknown = 0xFF12161E.toInt()
    for (j in 0 until h) {
        val row = (h - 1 - j) * w // map row 0 is the bottom
        for (i in 0 until w) {
            val c = cells[j * w + i].toInt()
            px[row + i] = when {
                c < 0 -> unknown
                else -> {
                    // free 0 -> #283041, occupied 100 -> #E8ECF3
                    val t = c.coerceAtMost(100) / 100f
                    val r = (0x28 + t * (0xE8 - 0x28)).toInt()
                    val g = (0x30 + t * (0xEC - 0x30)).toInt()
                    val b = (0x41 + t * (0xF3 - 0x41)).toInt()
                    (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
        }
    }
    val bmp = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    return Grid(bmp.asImageBitmap(), v[3], v[4], v[2], w, h)
}

/**
 * RViz-style view of Nav2: /map, the robot (TF map -> base_footprint, odom when there
 * is no map frame yet), /scan and the global /plan. Pinch/drag to move; in goal mode a
 * drag sets a goal and its heading, sent on /goal_pose like RViz's "2D Goal Pose".
 */
@Composable
fun MapView(modifier: Modifier = Modifier) {
    var grid by remember { mutableStateOf<Grid?>(null) }
    var scene by remember { mutableStateOf(MapScene("odom", null, emptyList(), emptyList())) }
    var mapCount by remember { mutableLongStateOf(0L) }
    var cx by remember { mutableFloatStateOf(0f) }
    var cy by remember { mutableFloatStateOf(0f) }
    var scale by remember { mutableFloatStateOf(60f) } // px per metre
    var follow by remember { mutableStateOf(true) }
    var goalMode by remember { mutableStateOf(false) }
    var goalDrag by remember { mutableStateOf<Pair<Offset, Offset>?>(null) }
    var goal by remember { mutableStateOf<Triple<Double, Double, Double>?>(null) }

    DisposableEffect(Unit) {
        onDispose { listOf(MAP, SCAN, PLAN, "/tf", "/tf_static").forEach { NativeLink.unsubscribe(it) } }
    }
    LaunchedEffect(Unit) {
        while (true) {
            NativeLink.subscribe(MAP, MsgType.OCCUPANCY_GRID, reliable = true, transientLocal = true)
            NativeLink.subscribe("/tf_static", MsgType.TF_MESSAGE, reliable = true, transientLocal = true)
            NativeLink.subscribe("/tf", MsgType.TF_MESSAGE, reliable = false)
            NativeLink.subscribe(SCAN, MsgType.LASER_SCAN, reliable = false)
            NativeLink.subscribe(PLAN, MsgType.PATH)
            NativeLink.advertise(GOAL, MsgType.POSE_STAMPED)

            val n = NativeLink.latestCount(MAP)
            if (n != mapCount) {
                val v = NativeLink.latestValues(MAP)
                val cells = NativeLink.latestBytes(MAP)
                if (v != null && cells != null && cells.size == v[0].toInt() * v[1].toInt() && cells.isNotEmpty()) {
                    grid = withContext(Dispatchers.Default) { gridBitmap(v, cells) }
                }
                mapCount = n
            }

            val tf = TfTree.parse(NativeLink.tfSnapshot())
            val fixed = if (tf.lookup("map", "base_footprint") != null) "map" else "odom"
            val robot = tf.lookup(fixed, "base_footprint")
            val scan = mutableListOf<Vec3>()
            NativeLink.latestValues(SCAN)?.let { s ->
                val frame = NativeLink.latestText(SCAN) ?: "base_footprint"
                val toFixed = tf.lookup(fixed, frame)
                if (toFixed != null && s.size > 4) {
                    for (k in 4 until s.size) {
                        val r = s[k]
                        if (!r.isFinite() || r < s[2] || r > s[3]) continue
                        val a = s[0] + (k - 4) * s[1]
                        scan += toFixed.apply(Vec3(r * cos(a), r * sin(a), 0.0))
                    }
                }
            }
            val plan = mutableListOf<Vec3>()
            NativeLink.latestValues(PLAN)?.let { p ->
                val toFixed = tf.lookup(fixed, NativeLink.latestText(PLAN) ?: "map") ?: Pose3.IDENTITY
                for (k in 0 until p.size / 2) plan += toFixed.apply(Vec3(p[2 * k], p[2 * k + 1], 0.0))
            }
            scene = MapScene(fixed, robot, scan, plan)
            if (follow && robot != null) {
                cx = robot.t.x.toFloat()
                cy = robot.t.y.toFloat()
            }
            delay(66)
        }
    }

    Box(modifier.clip(RoundedCornerShape(18.dp)).background(Palette.Bg)) {
        Canvas(
            Modifier.fillMaxSize().pointerInput(goalMode) {
                if (goalMode) {
                    awaitEachGesture {
                        val down = awaitFirstDown()
                        var end = down.position
                        goalDrag = down.position to end
                        while (true) {
                            val ch = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
                            if (!ch.pressed) break
                            end = ch.position
                            goalDrag = down.position to end
                            ch.consume()
                        }
                        goalDrag = null
                        val sx = size.width / 2f
                        val sy = size.height / 2f
                        val wx = cx + (down.position.x - sx) / scale
                        val wy = cy - (down.position.y - sy) / scale
                        val d = end - down.position
                        val heading = if (d.getDistance() > 20f) atan2(-d.y, d.x).toDouble()
                        else scene.robot?.let { atan2(wy - it.t.y, wx - it.t.x) } ?: 0.0
                        NativeLink.publishPose2d(GOAL, scene.fixed, wx.toDouble(), wy.toDouble(), heading)
                        goal = Triple(wx.toDouble(), wy.toDouble(), heading)
                        goalMode = false
                    }
                } else {
                    detectTransformGestures { _, pan, zoom, _ ->
                        if (pan != Offset.Zero) follow = false
                        scale = (scale * zoom).coerceIn(8f, 400f)
                        cx -= pan.x / scale
                        cy += pan.y / scale
                    }
                }
            },
        ) {
            val view = View(cx.toDouble(), cy.toDouble(), scale.toDouble(), size.width / 2.0, size.height / 2.0)
            grid?.let { drawGrid(view, it) }
            drawPlan(view, scene.plan)
            drawScan(view, scene.scan)
            goal?.let { (x, y, a) -> drawArrow(view.px(x, y), a, Palette.Warn, 26f) }
            scene.robot?.let { drawRobot(view, it) }
            goalDrag?.let { (a, b) ->
                drawCircle(Palette.Warn, 10f, a)
                drawLine(Palette.Warn, a, b, 5f, StrokeCap.Round)
            }
        }

        Row(Modifier.align(Alignment.TopEnd).padding(10.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Chip("Follow", follow, Palette.Accent) { follow = !follow }
            Chip(if (goalMode) "Drag to set goal" else "Goal", goalMode, Palette.Warn) { goalMode = !goalMode }
        }
        Column(Modifier.align(Alignment.BottomStart).padding(12.dp)) {
            val g = grid
            Text(
                if (g == null) "No /map yet — start navigation (launcher [N])"
                else "${g.w}×${g.h} @ ${"%.2f".format(g.res)} m · frame ${scene.fixed}",
                color = Palette.Muted, fontSize = 11.sp, fontFamily = Mono,
            )
            if (scene.robot == null) Text("No TF to base_footprint", color = Palette.Warn, fontSize = 11.sp)
        }
    }
}

@Composable
private fun Chip(text: String, on: Boolean, color: Color, onClick: () -> Unit) =
    Box(
        Modifier.clip(RoundedCornerShape(50)).background(if (on) color else Palette.Surface.copy(alpha = 0.9f))
            .clickable(onClick = onClick).padding(horizontal = 14.dp, vertical = 8.dp),
    ) { Text(text, color = if (on) Palette.Bg else Palette.Text, fontSize = 13.sp, fontWeight = FontWeight.SemiBold) }

private class View(val cx: Double, val cy: Double, val scale: Double, val sx: Double, val sy: Double) {
    fun px(x: Double, y: Double) = Offset((sx + (x - cx) * scale).toFloat(), (sy - (y - cy) * scale).toFloat())
}

private fun DrawScope.drawGrid(v: View, g: Grid) {
    val tl = v.px(g.ox, g.oy + g.h * g.res)
    drawImage(
        g.image,
        dstOffset = IntOffset(tl.x.roundToInt(), tl.y.roundToInt()),
        dstSize = IntSize((g.w * g.res * v.scale).roundToInt(), (g.h * g.res * v.scale).roundToInt()),
        filterQuality = FilterQuality.None,
    )
}

private fun DrawScope.drawPlan(v: View, plan: List<Vec3>) {
    if (plan.size < 2) return
    drawPoints(plan.map { v.px(it.x, it.y) }, PointMode.Polygon, Palette.Ok, 5f, StrokeCap.Round)
}

private fun DrawScope.drawScan(v: View, scan: List<Vec3>) {
    if (scan.isEmpty()) return
    drawPoints(scan.map { v.px(it.x, it.y) }, PointMode.Points, Palette.Danger, 5f, StrokeCap.Round)
}

private fun DrawScope.drawArrow(at: Offset, yaw: Double, color: Color, len: Float) {
    val dir = Offset(cos(yaw).toFloat(), -sin(yaw).toFloat())
    val side = Offset(-dir.y, dir.x)
    val tip = at + dir * len
    val path = Path().apply {
        moveTo(tip.x, tip.y)
        (at + side * (len * 0.45f) - dir * (len * 0.2f)).let { lineTo(it.x, it.y) }
        (at - side * (len * 0.45f) - dir * (len * 0.2f)).let { lineTo(it.x, it.y) }
        close()
    }
    drawPath(path, color)
}

private fun DrawScope.drawRobot(v: View, p: Pose3) {
    val yaw = p.q.yaw()
    val c = cos(yaw)
    val s = sin(yaw)
    // The costmap footprint from config/nav2.yaml.
    val corners = listOf(0.38 to 0.22, 0.38 to -0.22, -0.38 to -0.22, -0.38 to 0.22).map { (x, y) ->
        v.px(p.t.x + c * x - s * y, p.t.y + s * x + c * y)
    }
    val path = Path().apply {
        moveTo(corners[0].x, corners[0].y)
        corners.drop(1).forEach { lineTo(it.x, it.y) }
        close()
    }
    drawPath(path, Palette.Accent.copy(alpha = 0.25f))
    drawPath(path, Palette.Accent, style = Stroke(3f))
    drawArrow(v.px(p.t.x, p.t.y), yaw, Palette.Accent, maxOf(18f, (0.3 * v.scale).toFloat()))
}
