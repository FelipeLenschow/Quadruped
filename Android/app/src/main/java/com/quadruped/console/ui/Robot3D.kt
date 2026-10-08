package com.quadruped.console.ui

import android.opengl.GLSurfaceView
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
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
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.quadruped.console.MsgType
import com.quadruped.console.NativeLink
import com.quadruped.console.Pose3
import com.quadruped.console.Quat
import com.quadruped.console.TfTree
import com.quadruped.console.Urdf
import com.quadruped.console.Vec3
import com.quadruped.console.VoxelMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.math.cos
import kotlin.math.sin

private const val JOINTS = "/sensors/joint_states"
const val LIO_SNAPSHOT = "/lio/map_voxels"
const val LIO_DELTA = "/lio/map_voxels/delta"
const val LIO_VOXEL_SIZE = "/lio/map_voxels/size"
const val LIO_ODOM = "/lio/odom"

/** Nothing for this long and the mapper is taken to be off; it is optional. */
private const val NOT_RUNNING_MS = 15_000L

/**
 * The Go2 URDF, legs from /sensors/joint_states, standing in Point-LIO's voxel map.
 * With /lio/odom arriving the base sits at that pose in lio_odom, on the map; without it,
 * /tf (odom -> base) places it over a grid. /tf is only taken in that case, and the raw
 * clouds never: the weak Wi-Fi stays free for the heartbeat.
 * One finger orbits, two pan and pinch; panning stops following the robot.
 */
@Composable
fun Robot3D(modifier: Modifier = Modifier) {
    val ctx = LocalContext.current
    val urdf = remember { ctx.assets.open("go2/go2.urdf").use { Urdf.parse(it) } }
    val camera = remember { OrbitCamera().apply { dist = 2.5f; pitch = 0.6f } }
    val renderer = remember { RobotRenderer(ctx.assets, urdf, camera) }
    val map = remember { VoxelMap() }
    var glView by remember { mutableStateOf<GLSurfaceView?>(null) }
    var follow by remember { mutableStateOf(true) }
    var base by remember { mutableStateOf(Pose3(Vec3(0.0, 0.0, 0.33), Quat.IDENTITY)) }
    var jointsLive by remember { mutableStateOf(false) }
    var onLio by remember { mutableStateOf(false) }
    var voxels by remember { mutableIntStateOf(0) }
    var lastDeltaMs by remember { mutableLongStateOf(-1L) }
    var lastMapMs by remember { mutableLongStateOf(-1L) }
    val startedMs = remember { System.currentTimeMillis() }
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }

    DisposableEffect(Unit) {
        onDispose {
            for (t in listOf(JOINTS, "/tf", LIO_SNAPSHOT, LIO_DELTA, LIO_VOXEL_SIZE, LIO_ODOM)) NativeLink.unsubscribe(t)
        }
    }
    LaunchedEffect(Unit) {
        val trail = ArrayDeque<Vec3>()
        var snapshotCount = 0L
        withContext(Dispatchers.Default) {
            while (true) {
                // Idempotent; also re-subscribes after a reconnect. The latched snapshot
                // arrives again on every subscribe, so leaving the view costs nothing.
                NativeLink.subscribe(JOINTS, MsgType.JOINT_STATE, reliable = false)
                NativeLink.subscribe(LIO_SNAPSHOT, MsgType.POINT_CLOUD2, reliable = true, transientLocal = true)
                NativeLink.subscribe(LIO_DELTA, MsgType.POINT_CLOUD2, reliable = true, queue = true)
                NativeLink.subscribe(LIO_ODOM, MsgType.ODOMETRY, reliable = false)
                NativeLink.subscribe(LIO_VOXEL_SIZE, MsgType.FLOAT32, reliable = true, transientLocal = true)
                val t = System.currentTimeMillis()

                NativeLink.latestValues(LIO_VOXEL_SIZE)?.firstOrNull()?.takeIf { it > 0.0 && it != map.voxel }?.let {
                    map.setVoxel(it)
                    snapshotCount = 0L
                }

                val n = NativeLink.latestCount(LIO_SNAPSHOT)
                if (n != snapshotCount) {
                    snapshotCount = n
                    NativeLink.latestBytes(LIO_SNAPSHOT)?.let { map.replace(VoxelMap.floats(it)) }
                    if (n > 0) lastMapMs = t
                }
                NativeLink.takeQueued(LIO_DELTA)?.let {
                    map.add(VoxelMap.floats(it))
                    lastDeltaMs = t
                    lastMapMs = t
                }

                val lio = NativeLink.latestValues(LIO_ODOM)?.takeIf { it.size >= 7 && NativeLink.latestAgeMs(LIO_ODOM) in 0..1000 }
                if (lio != null) {
                    NativeLink.unsubscribe("/tf")
                    base = Pose3(Vec3(lio[0], lio[1], lio[2]), Quat(lio[3], lio[4], lio[5], lio[6]))
                } else {
                    NativeLink.subscribe("/tf", MsgType.TF_MESSAGE, reliable = false)
                    TfTree.parse(NativeLink.tfSnapshot()).lookup("odom", "base")?.let { base = it }
                }
                if (onLio != (lio != null)) {
                    // Different frames: the old trail would be drawn in the wrong place.
                    onLio = lio != null
                    trail.clear()
                }
                if (trail.isEmpty() || (trail.last() - base.t).norm() > 0.02) {
                    trail.addLast(base.t)
                    if (trail.size > 2000) trail.removeFirst()
                }

                val names = NativeLink.latestText(JOINTS)?.split(",")
                val q = NativeLink.latestValues(JOINTS)
                val joints = if (names != null && q != null) names.zip(q.toList()).toMap() else emptyMap()
                jointsLive = joints.isNotEmpty()
                val links = urdf.linkPoses(joints.ifEmpty { STAND }).mapValues { (_, p) -> base * p }
                renderer.scene = RobotScene(base, links, trail.toList(), grid = !onLio, cloud = map)
                voxels = map.size
                now = t
                glView?.requestRender()
                delay(33)
            }
        }
    }

    Box(modifier) {
        GlView(renderer) { glView = it }
        // On top of the GL view, which would otherwise take the touches.
        Box(
            Modifier.fillMaxSize().pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val e = awaitPointerEvent()
                        val fingers = e.changes.count { it.pressed }
                        val pan = e.calculatePan()
                        if (fingers >= 2) {
                            if (follow) {
                                follow = false
                                camera.focus = base.t
                            }
                            val y = camera.yaw.toDouble()
                            val k = camera.dist * 0.0015
                            val right = Vec3(-sin(y), cos(y), 0.0)
                            val ahead = Vec3(-cos(y), -sin(y), 0.0)
                            camera.focus = (camera.focus ?: base.t) - right * (pan.x * k) + ahead * (pan.y * k)
                            camera.dist = (camera.dist / e.calculateZoom()).coerceIn(0.6f, 80f)
                        } else if (fingers == 1) {
                            camera.yaw -= pan.x * 0.008f
                            camera.pitch = (camera.pitch + pan.y * 0.006f).coerceIn(-0.2f, 1.5f)
                        }
                        e.changes.forEach { if (it.positionChanged()) it.consume() }
                        glView?.requestRender()
                    } while (e.changes.any { it.pressed })
                }
            },
        )

        Column(
            Modifier.align(Alignment.TopStart).padding(10.dp).clip(RoundedCornerShape(10.dp))
                .background(Palette.Bg.copy(alpha = 0.7f)).padding(horizontal = 10.dp, vertical = 6.dp),
        ) {
            val quiet = if (lastMapMs < 0) now - startedMs else now - lastMapMs
            when {
                quiet > NOT_RUNNING_MS -> Text("3D map not running", color = Palette.Muted, fontSize = 12.sp)
                lastMapMs < 0 -> Text("Waiting for 3D map…", color = Palette.Muted, fontSize = 12.sp)
                else -> {
                    Text(
                        "%,d voxels".format(voxels) + if (voxels >= map.capacity) " (full)" else "",
                        color = Palette.Text, fontSize = 12.sp, fontFamily = Mono,
                    )
                    Text(
                        if (lastDeltaMs < 0) "no delta yet" else "delta ${(now - lastDeltaMs) / 1000}s ago",
                        color = Palette.Muted, fontSize = 12.sp, fontFamily = Mono,
                    )
                }
            }
            Text(if (onLio) "pose: $LIO_ODOM" else "pose: /tf", color = Palette.Muted, fontSize = 12.sp, fontFamily = Mono)
        }

        Box(
            Modifier.align(Alignment.TopEnd).padding(10.dp).clip(RoundedCornerShape(50))
                .background(if (follow) Palette.Accent else Palette.Surface2)
                .clickable {
                    follow = !follow
                    camera.focus = if (follow) null else base.t
                    glView?.requestRender()
                }
                .padding(horizontal = 12.dp, vertical = 7.dp),
        ) { Text("Follow", color = if (follow) Palette.Bg else Palette.Text, fontSize = 12.sp) }

        if (!jointsLive) {
            Text(
                "Waiting for $JOINTS",
                Modifier.align(Alignment.BottomCenter).padding(10.dp),
                color = Palette.Warn, fontSize = 11.sp,
            )
        }
    }
}

/** Go2's standing joints, until /sensors/joint_states arrives. */
private val STAND = buildMap {
    for (leg in listOf("FL", "FR", "RL", "RR")) {
        put("${leg}_hip_joint", if (leg.endsWith("L")) 0.1 else -0.1)
        put("${leg}_thigh_joint", if (leg.startsWith("F")) 0.8 else 1.0)
        put("${leg}_calf_joint", -1.5)
    }
}

/** The renderer's GL surface, paused with the app; [onView] hands it out for requestRender(). */
@Composable
fun GlView(renderer: RobotRenderer, onView: (GLSurfaceView) -> Unit) {
    var glView by remember { mutableStateOf<GLSurfaceView?>(null) }
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(lifecycle, glView) {
        val obs = LifecycleEventObserver { _, e ->
            when (e) {
                Lifecycle.Event.ON_PAUSE -> glView?.onPause()
                Lifecycle.Event.ON_RESUME -> glView?.onResume()
                else -> {}
            }
        }
        lifecycle.addObserver(obs)
        onDispose { lifecycle.removeObserver(obs) }
    }
    AndroidView(
        factory = {
            GLSurfaceView(it).apply {
                setEGLContextClientVersion(3)
                setEGLConfigChooser(MsaaConfigChooser())
                setRenderer(renderer)
                renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
                glView = this
                onView(this)
            }
        },
        modifier = Modifier.fillMaxSize(),
    )
}
