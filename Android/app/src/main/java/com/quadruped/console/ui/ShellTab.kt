package com.quadruped.console.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.view.KeyEvent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.quadruped.console.ShellSettings
import com.quadruped.console.ssh.ShellSession
import com.quadruped.console.ssh.SshKeys
import com.quadruped.console.ssh.SshTarget
import com.quadruped.console.ssh.SshTerminal
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * A real terminal on the robot over SSH, to start the driver and Nav2 without a laptop.
 * Independent of the console's DDS link: heartbeat and e-stop never go through here.
 */
@Composable
fun ShellTab(modifier: Modifier = Modifier) {
    val ctx = LocalContext.current
    var term by remember { mutableStateOf(ShellSession.current) }
    var state by remember { mutableStateOf(term?.state) }
    var message by remember { mutableStateOf(term?.message ?: "") }
    LaunchedEffect(term) {
        while (true) {
            state = term?.state
            message = term?.message ?: ""
            delay(250)
        }
    }

    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        val t = term
        if (t == null || state == SshTerminal.State.CLOSED) {
            ConnectPanel(lastMessage = if (t != null) message else null) { target, password ->
                t?.close()
                val next = SshTerminal.android(ctx, target, password)
                ShellSession.current = next
                term = next
            }
        } else {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.padding(end = 8.dp).width(8.dp).height(8.dp).clip(RoundedCornerShape(50))
                        .background(if (state == SshTerminal.State.CONNECTED) Palette.Ok else Palette.Warn),
                )
                Text(message, Modifier.weight(1f), color = Palette.Muted, fontSize = 12.sp, fontFamily = Mono, maxLines = 1)
                TextButton({
                    t.close()
                    ShellSession.current = null
                    term = null
                }) { Text("Disconnect", color = Palette.Muted) }
            }
            TerminalWithKeys(t, Modifier.weight(1f))
        }
    }
}

@Composable
private fun ConnectPanel(lastMessage: String?, onConnect: (SshTarget, String?) -> Unit) {
    val ctx = LocalContext.current
    val saved = remember { ShellSettings.load(ctx) }
    var host by remember { mutableStateOf(saved.host) }
    var port by remember { mutableStateOf(saved.port.toString()) }
    var user by remember { mutableStateOf(saved.user) }
    var password by remember { mutableStateOf("") }
    var shortcuts by remember { mutableStateOf(saved.shortcuts) }
    var publicKey by remember { mutableStateOf("") }
    LaunchedEffect(Unit) {
        publicKey = withContext(Dispatchers.IO) { SshKeys(ctx.filesDir.resolve("ssh")).publicLine() }
    }

    Column(
        Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).clip(RoundedCornerShape(18.dp))
            .background(Palette.Surface).border(1.dp, Palette.Outline.copy(alpha = 0.6f), RoundedCornerShape(18.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Robot shell (SSH)", fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
        if (lastMessage != null) Text(lastMessage, color = Palette.Warn, fontSize = 12.sp, fontFamily = Mono)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(host, { host = it }, Modifier.weight(2f), label = { Text("Host") }, singleLine = true)
            OutlinedTextField(
                port, { port = it }, Modifier.weight(1f), label = { Text("Port") }, singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            )
        }
        OutlinedTextField(user, { user = it }, Modifier.fillMaxWidth(), label = { Text("User") }, singleLine = true)
        OutlinedTextField(
            password, { password = it }, Modifier.fillMaxWidth(), singleLine = true,
            label = { Text("Password (first time only)") }, visualTransformation = PasswordVisualTransformation(),
        )
        Text(
            "With a password, the app adds its own key to ~/.ssh/authorized_keys, so later logins need none.",
            color = Palette.Muted, fontSize = 12.sp,
        )
        Button(
            {
                val s = ShellSettings(host.trim(), port.toIntOrNull() ?: 22, user.trim(), shortcuts)
                ShellSettings.save(ctx, s)
                onConnect(SshTarget(s.host, s.port, s.user), password.ifEmpty { null })
                password = ""
            },
            Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.Bg),
        ) { Text("Connect", fontWeight = FontWeight.Bold) }

        Text("SHORTCUTS  (label = command, typed without Enter)", fontSize = 11.sp, color = Palette.Muted)
        OutlinedTextField(
            shortcuts, { shortcuts = it }, Modifier.fillMaxWidth(),
            textStyle = androidx.compose.ui.text.TextStyle(fontFamily = Mono, fontSize = 12.sp, color = Palette.Text),
        )
        Text(
            "Run launches inside tmux: if the phone sleeps or Wi-Fi drops, the SSH session ends and " +
                "anything started outside tmux (the driver!) dies with it.",
            color = Palette.Warn, fontSize = 12.sp,
        )
        Text("APP PUBLIC KEY", fontSize = 11.sp, color = Palette.Muted)
        Text(
            publicKey, fontFamily = Mono, fontSize = 10.sp, color = Palette.Muted,
            modifier = Modifier.clickable {
                ctx.getSystemService(ClipboardManager::class.java)
                    .setPrimaryClip(ClipData.newPlainText("ssh key", publicKey))
            },
        )
    }
}

@Composable
private fun TerminalWithKeys(t: SshTerminal, modifier: Modifier) {
    val ctx = LocalContext.current
    val shortcuts = remember { ShellSettings.load(ctx).shortcutList() }
    var view by remember { mutableStateOf<TerminalCanvasView?>(null) }
    var ctrl by remember { mutableStateOf(false) }

    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        shortcuts.forEach { (label, cmd) ->
            Box(
                Modifier.clip(RoundedCornerShape(50)).background(Palette.Surface2).clickable { view?.sendText(cmd) }
                    .padding(horizontal = 12.dp, vertical = 7.dp),
            ) { Text(label, fontSize = 12.sp, color = Palette.Text) }
        }
    }
    Box(modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Palette.Bg)) {
        AndroidView(
            factory = { c ->
                TerminalCanvasView(c).also {
                    it.onCtrlConsumed = { ctrl = false }
                    view = it
                }
            },
            update = { v ->
                if (v.terminal !== t) v.terminal = t
                v.ctrlArmed = ctrl
            },
            modifier = Modifier.fillMaxSize().padding(4.dp),
        )
    }
    ExtraKeys(ctrl, onCtrl = { ctrl = !ctrl }, view = view)
}

/** Two rows that fit the width: tmux in one tap, then the keys a phone keyboard lacks. */
@Composable
private fun ExtraKeys(ctrl: Boolean, onCtrl: () -> Unit, view: TerminalCanvasView?) {
    @Composable
    fun RowScope.key(label: String, active: Boolean = false, color: Color = Palette.Text, onClick: () -> Unit) =
        Box(
            Modifier.weight(1f).clip(RoundedCornerShape(8.dp)).background(if (active) Palette.Accent else Palette.Surface2)
                .clickable(onClick = onClick).padding(vertical = 9.dp),
            contentAlignment = Alignment.Center,
        ) { Text(label, color = if (active) Palette.Bg else color, fontSize = 13.sp, fontFamily = Mono, maxLines = 1) }

    fun tmux(keys: String) = view?.sendText("\u0002$keys")

    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            key("+win", color = Palette.Accent) { tmux("c") }
            for (i in 0..3) key("$i", color = Palette.Accent) { tmux("$i") }
            key("‹", color = Palette.Accent) { tmux("p") }
            key("›", color = Palette.Accent) { tmux("n") }
            key("^B", color = Palette.Accent) { view?.sendText("\u0002") }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            key("ESC") { view?.sendKey(KeyEvent.KEYCODE_ESCAPE) }
            key("TAB") { view?.sendKey(KeyEvent.KEYCODE_TAB) }
            key("CTRL", active = ctrl, onClick = onCtrl)
            key("^C", color = Palette.Danger) { view?.sendText("\u0003") }
            key("←") { view?.sendKey(KeyEvent.KEYCODE_DPAD_LEFT) }
            key("↑") { view?.sendKey(KeyEvent.KEYCODE_DPAD_UP) }
            key("↓") { view?.sendKey(KeyEvent.KEYCODE_DPAD_DOWN) }
            key("→") { view?.sendKey(KeyEvent.KEYCODE_DPAD_RIGHT) }
            key("⏎") { view?.sendText("\r") }
            key("⌨") { view?.showKeyboard() }
        }
    }
}
