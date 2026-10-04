package com.quadruped.console.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

object Palette {
    val Bg = Color(0xFF0B0D12)
    val Surface = Color(0xFF151922)
    val Surface2 = Color(0xFF1D2230)
    val Outline = Color(0xFF2A3140)
    val Text = Color(0xFFE8ECF3)
    val Muted = Color(0xFF8A93A6)
    val Accent = Color(0xFF3DD6C6)
    val Ok = Color(0xFF3DDC84)
    val Warn = Color(0xFFF5B544)
    val Danger = Color(0xFFFF4D5E)
    val DangerDeep = Color(0xFFB3182B)
}

val Mono = FontFamily.Monospace

private val scheme = darkColorScheme(
    primary = Palette.Accent,
    onPrimary = Palette.Bg,
    secondary = Palette.Warn,
    background = Palette.Bg,
    onBackground = Palette.Text,
    surface = Palette.Surface,
    onSurface = Palette.Text,
    surfaceVariant = Palette.Surface2,
    onSurfaceVariant = Palette.Muted,
    surfaceContainer = Palette.Surface,
    outline = Palette.Outline,
    error = Palette.Danger,
)

private val type = Typography(
    titleLarge = TextStyle(fontWeight = FontWeight.Bold, fontSize = 20.sp, letterSpacing = 0.5.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp),
    labelSmall = TextStyle(fontWeight = FontWeight.Medium, fontSize = 11.sp, letterSpacing = 1.2.sp),
)

@Composable
fun ConsoleTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = scheme, typography = type, content = content)
