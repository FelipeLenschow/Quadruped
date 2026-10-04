package com.quadruped.console.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Touch stick reporting x (right +) and y (down +) in -1..1. It springs back to zero
 * on release and also when it leaves the screen or is disabled mid-drag, so a stuck
 * finger state can never keep the robot walking.
 */
@Composable
fun Joystick(
    enabled: Boolean,
    horizontalOnly: Boolean,
    onMove: (x: Float, y: Float) -> Unit,
    modifier: Modifier = Modifier,
    diameter: Dp = 168.dp,
) {
    var knob by remember { mutableStateOf(Offset.Zero) }
    val move by rememberUpdatedState(onMove)

    LaunchedEffect(enabled) { if (!enabled) knob = Offset.Zero }
    DisposableEffect(Unit) { onDispose { move(0f, 0f) } }

    Box(
        modifier.size(diameter).pointerInput(enabled, horizontalOnly) {
            if (!enabled) return@pointerInput
            awaitEachGesture {
                val down = awaitFirstDown()
                val r = size.width / 2f
                fun update(p: Offset) {
                    var v = (p - Offset(r, r)) / r
                    if (horizontalOnly) v = Offset(v.x, 0f)
                    val len = v.getDistance()
                    if (len > 1f) v /= len
                    knob = v
                    move(v.x, v.y)
                }
                try {
                    update(down.position)
                    while (true) {
                        val ch = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
                        if (!ch.pressed) break
                        update(ch.position)
                        ch.consume()
                    }
                } finally {
                    knob = Offset.Zero
                    move(0f, 0f)
                }
            }
        },
    ) {
        Canvas(Modifier.fillMaxSize()) {
            val r = size.minDimension / 2f
            val c = center
            val ring = if (enabled) Palette.Accent else Palette.Outline
            drawCircle(Brush.radialGradient(listOf(Palette.Surface2, Palette.Surface), c, r), r, c)
            drawCircle(ring.copy(alpha = 0.55f), r - 2f, c, style = Stroke(3f))
            drawCircle(Palette.Outline, r * 0.5f, c, style = Stroke(2f))
            if (horizontalOnly) {
                drawRoundRect(
                    Palette.Outline.copy(alpha = 0.6f),
                    topLeft = Offset(c.x - r * 0.8f, c.y - 6f), size = Size(r * 1.6f, 12f),
                    cornerRadius = androidx.compose.ui.geometry.CornerRadius(6f),
                )
            } else {
                drawLine(Palette.Outline, Offset(c.x, c.y - r * 0.8f), Offset(c.x, c.y + r * 0.8f), 2f, StrokeCap.Round)
                drawLine(Palette.Outline, Offset(c.x - r * 0.8f, c.y), Offset(c.x + r * 0.8f, c.y), 2f, StrokeCap.Round)
            }
            val k = c + knob * (r * 0.68f)
            val kr = r * 0.3f
            if (knob != Offset.Zero) drawLine(ring.copy(alpha = 0.5f), c, k, 6f, StrokeCap.Round)
            drawCircle(
                Brush.radialGradient(
                    if (enabled) listOf(Palette.Accent, Palette.Accent.copy(alpha = 0.6f))
                    else listOf(Palette.Outline, Palette.Surface2),
                    k, kr,
                ),
                kr, k,
            )
        }
    }
}
