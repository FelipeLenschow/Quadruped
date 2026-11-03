package com.quadruped.console.ui

import android.content.Context
import android.graphics.BitmapFactory
import android.net.ConnectivityManager
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.quadruped.console.RobotNetwork
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

// llm_host --app (launcher [A]) on the robot's AP address; res/xml/network_security_config.xml allows plain HTTP
// to it, and to nothing else.
private const val CHAT_URL = "http://10.42.0.1:8766"

private data class ChatEvent(val seq: Int, val kind: String, val text: String, val image: String?)

/** The chat, kept across tab switches. */
private object ChatStore {
    var chat = -1
    val events = mutableStateListOf<ChatEvent>()
    var status by mutableStateOf("")
    var online by mutableStateOf(false)
    val images = mutableStateMapOf<String, ImageBitmap>()
}

/** Through the robot's Wi-Fi: with no internet there, Android may keep mobile data as the default network. */
private fun connect(ctx: Context, path: String, timeoutMs: Int): HttpURLConnection {
    val url = URL(CHAT_URL + path)
    val network = RobotNetwork.pick(ctx.getSystemService(ConnectivityManager::class.java))?.first
    return ((network?.openConnection(url) ?: url.openConnection()) as HttpURLConnection).apply {
        connectTimeout = 3000
        readTimeout = timeoutMs
    }
}

private fun httpGet(ctx: Context, path: String, timeoutMs: Int): ByteArray {
    val c = connect(ctx, path, timeoutMs)
    try {
        if (c.responseCode != 200) error("HTTP ${c.responseCode}")
        return c.inputStream.use { it.readBytes() }
    } finally {
        c.disconnect()
    }
}

private fun httpPost(ctx: Context, path: String, body: JSONObject = JSONObject()) {
    val c = connect(ctx, path, 5000)
    try {
        c.requestMethod = "POST"
        c.doOutput = true
        c.setRequestProperty("Content-Type", "application/json")
        c.outputStream.use { it.write(body.toString().toByteArray()) }
        c.responseCode
    } finally {
        c.disconnect()
    }
}

/**
 * The LLM agent's chat (launcher [A]). The first message starts a chat with it as the task; later ones answer
 * the agent, or reach it before its next step while it works. Stop stops the chat and the robot; New chat
 * also clears it. Photos (look's, and find's with its box drawn) show in the list; a tap shows one full size.
 */
@Composable
fun ChatTab(modifier: Modifier) {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var draft by rememberSaveable { mutableStateOf("") }
    var full by remember { mutableStateOf<ImageBitmap?>(null) }
    val list = rememberLazyListState()

    // Long polls: the robot answers at once when there is something new, else after 20 s.
    LaunchedEffect(Unit) {
        while (isActive) {
            val after = if (ChatStore.chat < 0) 0 else ChatStore.events.lastOrNull()?.seq ?: 0
            val reply = runCatching {
                withContext(Dispatchers.IO) { JSONObject(String(httpGet(ctx, "/chat?after=$after&wait=20", 25_000))) }
            }.getOrNull()
            if (reply == null) {
                ChatStore.online = false
                ChatStore.status = "no agent: start launcher [A] on the robot"
                delay(2000)
                continue
            }
            ChatStore.online = true
            ChatStore.status = reply.optString("status")
            val chat = reply.optInt("chat")
            if (chat != ChatStore.chat) {
                // A new chat (from here or another phone): its events start again at 1.
                ChatStore.chat = chat
                ChatStore.events.clear()
                if (after > 0) continue
            }
            val events = reply.optJSONArray("events") ?: continue
            for (i in 0 until events.length()) {
                val e = events.getJSONObject(i)
                val image = e.optString("image").ifEmpty { null }
                ChatStore.events.add(ChatEvent(e.getInt("seq"), e.optString("kind"), e.optString("text"), image))
                if (image != null && image !in ChatStore.images) {
                    scope.launch {
                        val bytes = runCatching { withContext(Dispatchers.IO) { httpGet(ctx, image, 10_000) } }.getOrNull()
                        bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }?.let {
                            ChatStore.images[image] = it.asImageBitmap()
                        }
                    }
                }
            }
        }
    }
    LaunchedEffect(ChatStore.events.size) {
        if (ChatStore.events.isNotEmpty()) list.animateScrollToItem(ChatStore.events.size - 1)
    }

    fun post(path: String, body: JSONObject = JSONObject()) {
        scope.launch { runCatching { withContext(Dispatchers.IO) { httpPost(ctx, path, body) } } }
    }

    val online = ChatStore.online
    val idle = ChatStore.status == "waiting for a task"
    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            val dot = when {
                !online -> Palette.Muted
                idle || ChatStore.status == "waiting for you" -> Palette.Ok
                else -> Palette.Warn
            }
            Box(Modifier.size(8.dp).clip(CircleShape).background(dot))
            Text(
                ChatStore.status.ifEmpty { "connecting…" }, Modifier.weight(1f), color = Palette.Muted, fontSize = 13.sp,
                maxLines = 2, overflow = TextOverflow.Ellipsis,
            )
            OutlinedButton(
                { post("/chat/stop") }, enabled = online && !idle, border = BorderStroke(1.dp, Palette.Danger),
            ) { Text("Stop", color = if (online && !idle) Palette.Danger else Palette.Muted) }
            OutlinedButton({
                post("/chat/new")
                ChatStore.chat = -1
                ChatStore.events.clear()
            }, enabled = online, border = BorderStroke(1.dp, Palette.Outline)) {
                Text("New chat", color = if (online) Palette.Text else Palette.Muted)
            }
        }
        LazyColumn(Modifier.fillMaxWidth().weight(1f), state = list, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (ChatStore.events.isEmpty()) {
                item {
                    Text(
                        if (online) "Send a task, e.g. \"find the shoe and pee on it\"." else
                            "The agent runs on the robot: launcher [A], with mcp:=true in [6] and Ollama running.",
                        color = Palette.Muted, fontSize = 13.sp, modifier = Modifier.padding(vertical = 12.dp),
                    )
                }
            }
            items(ChatStore.events, key = { it.seq }) { ChatRow(it) { bmp -> full = bmp } }
        }
        if (ChatStore.status == "waiting for you") {
            OutlinedButton({ post("/chat", JSONObject().put("text", "")) }, border = BorderStroke(1.dp, Palette.Outline)) {
                Text("Keep going", color = Palette.Text)
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                draft, { draft = it }, Modifier.weight(1f), enabled = online, maxLines = 4,
                placeholder = { Text(if (idle) "A task for the robot" else "Answer the agent", color = Palette.Muted) },
            )
            Button(
                {
                    post("/chat", JSONObject().put("text", draft.trim()))
                    draft = ""
                },
                enabled = online && draft.isNotBlank(),
                colors = ButtonDefaults.buttonColors(containerColor = Palette.Accent, contentColor = Palette.Bg),
            ) { Text("Send", fontWeight = FontWeight.Bold) }
        }
    }

    full?.let { bmp ->
        Dialog({ full = null }) {
            Image(bmp, null, Modifier.fillMaxWidth().clickable { full = null }, contentScale = ContentScale.FillWidth)
        }
    }
}

@Composable
private fun ChatRow(e: ChatEvent, onImage: (ImageBitmap) -> Unit) {
    when (e.kind) {
        "you" -> Row(Modifier.fillMaxWidth()) {
            Spacer(Modifier.weight(1f))
            Text(
                e.text.ifEmpty { "(keep going)" },
                Modifier.widthIn(max = 300.dp).clip(RoundedCornerShape(14.dp)).background(Palette.Accent)
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                color = Palette.Bg, fontSize = 14.sp,
            )
        }
        "said" -> Text(
            e.text,
            Modifier.widthIn(max = 320.dp).clip(RoundedCornerShape(14.dp)).background(Palette.Surface2)
                .padding(horizontal = 12.dp, vertical = 8.dp),
            color = Palette.Text, fontSize = 14.sp,
        )
        "call" -> Text("→ ${e.text}", color = Palette.Accent, fontSize = 12.sp, fontFamily = Mono)
        "info" -> Text(e.text, Modifier.fillMaxWidth(), color = Palette.Muted, fontSize = 12.sp)
        else -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            // result / error: a few lines, all on a tap.
            var open by remember { mutableStateOf(false) }
            Text(
                e.text, Modifier.clickable { open = !open },
                color = if (e.kind == "error") Palette.Danger else Palette.Muted, fontSize = 12.sp, fontFamily = Mono,
                maxLines = if (open) Int.MAX_VALUE else 3, overflow = TextOverflow.Ellipsis,
            )
            e.image?.let { path ->
                val bmp = ChatStore.images[path]
                if (bmp == null) {
                    Text("loading photo…", color = Palette.Muted, fontSize = 12.sp)
                } else {
                    Image(
                        bmp, null,
                        Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).clickable { onImage(bmp) },
                        contentScale = ContentScale.FillWidth,
                    )
                }
            }
        }
    }
}
