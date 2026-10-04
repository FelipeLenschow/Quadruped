package com.quadruped.console.ssh

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.ConnectivityManager
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.jcraft.jsch.SocketFactory
import com.quadruped.console.RobotNetwork
import com.termux.terminal.TerminalColors
import com.termux.terminal.TerminalEmulator
import com.termux.terminal.TerminalOutput
import com.termux.terminal.TerminalSession
import com.termux.terminal.TerminalSessionClient
import com.termux.terminal.TextStyle
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.net.Socket
import java.util.concurrent.Executors

/**
 * One SSH shell drawn by Termux's xterm emulator. The emulator is only touched through
 * `post` (the main thread on Android). Two threads of our own: one connects and then
 * reads for the life of the session, the other only writes, so keys never queue behind
 * the read loop. Lives in [ShellSession], not in the UI, so leaving the tab keeps the
 * shell. Launches belong inside tmux on the robot: a dropped session kills what it started.
 */
class SshTerminal(
    val target: SshTarget,
    private val password: String?,
    keysDir: File,
    private val post: (Runnable) -> Unit,
    private val socketFactory: () -> SocketFactory? = { null },
    private val clipboard: Clipboard? = null,
) : TerminalOutput() {
    enum class State { CONNECTING, CONNECTED, CLOSED }

    interface Clipboard {
        fun copy(text: String)
        fun paste(): String?
    }

    @Volatile var state = State.CONNECTING
        private set
    @Volatile var message = "Connecting to ${target.user}@${target.host}..."
        private set

    /** On the `post` thread: the screen changed. */
    var onUpdate: (() -> Unit)? = null

    val emulator: TerminalEmulator
    private val writer = Executors.newSingleThreadExecutor()
    private val client = SshClient(SshKeys(keysDir))
    @Volatile private var output: OutputStream? = null
    @Volatile private var cols = 80
    @Volatile private var rows = 24
    @Volatile private var pixels = 0 to 0

    init {
        theme()
        emulator = TerminalEmulator(this, cols, rows, 3000, Logs)
        emulator.setCursorBlinkingEnabled(false)
        emulator.setCursorBlinkState(true)
        local("${message}\r\n")
        Thread(::run, "ssh-${target.host}").apply { isDaemon = true }.start()
    }

    private fun run() {
        try {
            client.connect(target, password, socketFactory = socketFactory())
            if (!password.isNullOrEmpty()) {
                val (status, out) = client.installKey()
                local(
                    if (status == 0) "App key added to ~/.ssh/authorized_keys; next time no password.\r\n"
                    else "Could not add the app key: ${out.trim()}\r\n",
                )
            }
            val (input, out) = client.openShell(cols, rows)
            // The view may have measured itself while we were connecting.
            client.resize(cols, rows, pixels.first, pixels.second)
            output = out
            state = State.CONNECTED
            message = "${target.user}@${target.host}"
            post(Runnable { onUpdate?.invoke() })
            pump(input)
            finish("Connection closed.")
        } catch (e: Exception) {
            Log.w("SshTerminal", "ssh", e)
            finish("SSH failed: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private fun pump(input: InputStream) {
        val buf = ByteArray(8192)
        while (true) {
            val n = input.read(buf)
            if (n < 0) return
            val chunk = buf.copyOf(n)
            post(Runnable {
                emulator.append(chunk, chunk.size)
                onUpdate?.invoke()
            })
        }
    }

    @Synchronized
    private fun finish(why: String) {
        if (state == State.CLOSED) return
        state = State.CLOSED
        message = why
        local("\r\n\u001b[33m[$why]\u001b[0m\r\n")
        client.disconnect()
        writer.shutdown()
    }

    /** Text printed by the app itself, not the robot. */
    private fun local(text: String) {
        val b = text.toByteArray()
        post(Runnable {
            emulator.append(b, b.size)
            onUpdate?.invoke()
        })
    }

    fun send(text: String) {
        val b = text.toByteArray()
        write(b, 0, b.size)
    }

    /** Called by the emulator (its replies to the remote program) and by [send] (keys). */
    override fun write(data: ByteArray, offset: Int, count: Int) {
        val copy = data.copyOfRange(offset, offset + count)
        if (writer.isShutdown) return
        writer.execute {
            try {
                output?.apply {
                    write(copy)
                    flush()
                }
            } catch (e: Exception) {
                finish("Write failed: ${e.message}")
            }
        }
    }

    /** On the `post` thread, from the view's size. */
    fun resize(newCols: Int, newRows: Int, widthPx: Int, heightPx: Int) {
        if (newCols < 4 || newRows < 2 || (newCols == cols && newRows == rows)) return
        cols = newCols
        rows = newRows
        pixels = widthPx to heightPx
        emulator.resize(newCols, newRows)
        onUpdate?.invoke()
        if (writer.isShutdown) return
        writer.execute {
            try {
                client.resize(newCols, newRows, widthPx, heightPx)
            } catch (_: Exception) {
            }
        }
    }

    fun close() {
        Thread { finish("Disconnected.") }.start()
    }

    override fun titleChanged(oldTitle: String?, newTitle: String?) {}
    override fun onCopyTextToClipboard(text: String?) { if (text != null) clipboard?.copy(text) }
    override fun onPasteTextFromClipboard() { clipboard?.paste()?.takeIf { it.isNotEmpty() }?.let(emulator::paste) }
    override fun onBell() {}
    override fun onColorsChanged() {}

    private object Logs : TerminalSessionClient {
        override fun onTextChanged(s: TerminalSession) {}
        override fun onTitleChanged(s: TerminalSession) {}
        override fun onSessionFinished(s: TerminalSession) {}
        override fun onCopyTextToClipboard(s: TerminalSession, text: String?) {}
        override fun onPasteTextFromClipboard(s: TerminalSession?) {}
        override fun onBell(s: TerminalSession) {}
        override fun onColorsChanged(s: TerminalSession) {}
        override fun onTerminalCursorStateChange(state: Boolean) {}
        override fun getTerminalCursorStyle(): Int? = null
        override fun logError(tag: String?, message: String?) { Log.e(tag, message ?: "") }
        override fun logWarn(tag: String?, message: String?) { Log.w(tag, message ?: "") }
        override fun logInfo(tag: String?, message: String?) {}
        override fun logDebug(tag: String?, message: String?) {}
        override fun logVerbose(tag: String?, message: String?) {}
        override fun logStackTraceWithMessage(tag: String?, message: String?, e: Exception?) { Log.e(tag, message, e) }
        override fun logStackTrace(tag: String?, e: Exception?) { Log.e(tag, "", e) }
    }

    companion object {
        /** The app's palette as the terminal's default colours; survives a `reset` on the robot. */
        private fun theme() {
            val c = TerminalColors.COLOR_SCHEME.mDefaultColors
            c[TextStyle.COLOR_INDEX_FOREGROUND] = 0xFFE8ECF3.toInt()
            c[TextStyle.COLOR_INDEX_BACKGROUND] = 0xFF0B0D12.toInt()
            c[TextStyle.COLOR_INDEX_CURSOR] = 0xFF3DD6C6.toInt()
        }

        /** Main-thread emulator, sockets on the robot's Wi-Fi, the system clipboard. */
        fun android(ctx: Context, target: SshTarget, password: String?): SshTerminal {
            val app = ctx.applicationContext
            val main = Handler(Looper.getMainLooper())
            val clip = app.getSystemService(ClipboardManager::class.java)
            return SshTerminal(
                target, password, app.filesDir.resolve("ssh"),
                post = { main.post(it) },
                socketFactory = {
                    RobotNetwork.pick(app.getSystemService(ConnectivityManager::class.java))?.first?.let(::NetworkSockets)
                },
                clipboard = object : Clipboard {
                    override fun copy(text: String) = clip.setPrimaryClip(ClipData.newPlainText("terminal", text))
                    override fun paste() = clip.primaryClip?.getItemAt(0)?.coerceToText(app)?.toString()
                },
            )
        }
    }

    /** Network-bound sockets: SSH goes out on the robot's Wi-Fi even if mobile data is the default. */
    private class NetworkSockets(private val network: android.net.Network) : SocketFactory {
        override fun createSocket(host: String, port: Int): Socket = network.socketFactory.createSocket(host, port)
        override fun getInputStream(socket: Socket): InputStream = socket.getInputStream()
        override fun getOutputStream(socket: Socket): OutputStream = socket.getOutputStream()
    }
}

/** The one shell, kept across tab switches. */
object ShellSession {
    @Volatile var current: SshTerminal? = null
}
