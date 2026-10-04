package com.quadruped.console.ui

import android.opengl.GLSurfaceView
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
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
import kotlinx.coroutines.delay

private const val JOINTS = "/sensors/joint_states"

/**
 * The Go2 URDF posed live: /tf (odom -> base) places the base and /sensors/joint_states
 * drives the URDF's joints. Drag to orbit, pinch to zoom. The grid is fixed in odom,
 * so walking shows as the grid sliding under the robot, with a trail behind.
 */
@Composable
fun Robot3D(modifier: Modifier = Modifier) {
    val ctx = LocalContext.current
    val urdf = remember { ctx.assets.open("go2/go2.urdf").use { Urdf.parse(it) } }
    val camera = remember { OrbitCamera() }
    val renderer = remember { RobotRenderer(ctx.assets, urdf, camera) }
    var glView by remember { mutableStateOf<GLSurfaceView?>(null) }
    var live by remember { mutableStateOf(false) }
    val trail = remember { ArrayDeque<Vec3>() }

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
    DisposableEffect(Unit) {
        onDispose {
            NativeLink.unsubscribe(JOINTS)
            NativeLink.unsubscribe("/tf")
        }
    }
    LaunchedEffect(Unit) {
        while (true) {
            // Idempotent; also re-subscribes after a reconnect.
            NativeLink.subscribe(JOINTS, MsgType.JOINT_STATE, reliable = false)
            NativeLink.subscribe("/tf", MsgType.TF_MESSAGE, reliable = false)
            val names = NativeLink.latestText(JOINTS)?.split(",")
            val q = NativeLink.latestValues(JOINTS)
            val joints = if (names != null && q != null) names.zip(q.toList()).toMap() else emptyMap()
            live = joints.isNotEmpty()
            val base = TfTree.parse(NativeLink.tfSnapshot()).lookup("odom", "base")
                ?: Pose3(Vec3(0.0, 0.0, 0.33), Quat.IDENTITY)
            if (trail.isEmpty() || (trail.last() - base.t).norm() > 0.02) {
                trail.addLast(base.t)
                if (trail.size > 2000) trail.removeFirst()
            }
            val links = urdf.linkPoses(joints).mapValues { (_, p) -> base * p }
            renderer.scene = RobotScene(base, links, trail.toList())
            glView?.requestRender()
            delay(33)
        }
    }

    Box(modifier) {
        AndroidView(
            factory = {
                GLSurfaceView(it).apply {
                    setEGLContextClientVersion(3)
                    setEGLConfigChooser(MsaaConfigChooser())
                    setRenderer(renderer)
                    renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
                    glView = this
                }
            },
            modifier = Modifier.fillMaxSize(),
        )
        // On top of the GL view, which would otherwise take the touches.
        Box(
            Modifier.fillMaxSize().pointerInput(Unit) {
                detectTransformGestures { _, pan, zoom, _ ->
                    camera.yaw -= pan.x * 0.008f
                    camera.pitch = (camera.pitch + pan.y * 0.006f).coerceIn(-0.2f, 1.5f)
                    camera.dist = (camera.dist / zoom).coerceIn(0.6f, 6f)
                    glView?.requestRender()
                }
            },
        )
        if (!live) {
            Text(
                "Waiting for $JOINTS",
                Modifier.align(Alignment.BottomCenter).padding(10.dp),
                color = Palette.Warn, fontSize = 11.sp,
            )
        }
    }
}
