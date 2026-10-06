package com.quadruped.console.ssh

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import com.termux.terminal.TerminalEmulator
import com.termux.terminal.TerminalOutput
import java.io.OutputStream
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.io.File
import java.nio.file.Files

class SshClientTest {
    @Test
    fun installKeyCommandAppendsOnce() {
        val home = Files.createTempDirectory("home").toFile()
        val line = "ecdsa-sha2-nistp256 AAAAtest quadruped-console"
        repeat(2) {
            val p = ProcessBuilder("bash", "-c", installKeyCommand(line))
                .apply { environment()["HOME"] = home.path }.redirectErrorStream(true).start()
            assertEquals("installed", p.inputStream.bufferedReader().readText().trim())
            assertEquals(0, p.waitFor())
        }
        assertEquals(listOf(line), File(home, ".ssh/authorized_keys").readLines())
        home.deleteRecursively()
    }

    /**
     * Against a real sshd, only when SSH_TEST_PORT is set (see Android/README.md). The
     * app key must already be in that sshd's AuthorizedKeysFile: SSH_TEST_KEYS is the
     * key directory to use.
     */
    @Test
    fun keyLoginExecAndPtyShell() {
        val port = System.getenv("SSH_TEST_PORT")?.toIntOrNull()
        assumeTrue(port != null)
        val keys = SshKeys(File(System.getenv("SSH_TEST_KEYS")!!))
        val client = SshClient(keys)
        client.connect(SshTarget("127.0.0.1", port!!, System.getProperty("user.name")), null)
        assertEquals(0 to "hi\n", client.exec("echo hi"))

        val (input, output) = client.openShell(100, 30)
        output.write("echo TERM=\$TERM COLS=\$(tput cols); exit\n".toByteArray())
        output.flush()
        val text = input.bufferedReader().readText()
        assertTrue(text, text.contains("TERM=xterm-256color COLS=100"))
        client.disconnect()
    }

    /** tmux through SSH into Termux's emulator, its replies going back: what the Shell tab does. */
    @Test
    fun tmuxRendersInTheEmulator() {
        val port = System.getenv("SSH_TEST_PORT")?.toIntOrNull()
        assumeTrue(port != null)
        val client = SshClient(SshKeys(File(System.getenv("SSH_TEST_KEYS")!!)))
        client.connect(SshTarget("127.0.0.1", port!!, System.getProperty("user.name")), null)
        val (input, output) = client.openShell(80, 24)
        val emu = TerminalEmulator(Replies(output), 80, 24, 1000, null)
        val reader = Thread {
            val buf = ByteArray(8192)
            while (true) {
                val n = try { input.read(buf) } catch (_: Exception) { -1 }
                if (n < 0) break
                synchronized(emu) { emu.append(buf, n) }
            }
        }.apply { isDaemon = true; start() }

        fun type(s: String) = output.apply { write(s.toByteArray()); flush() }
        fun screen() = synchronized(emu) { emu.screen.getSelectedText(0, 0, emu.mColumns - 1, emu.mRows - 1) }
        fun waitFor(text: String): String {
            val end = System.currentTimeMillis() + 8000
            while (System.currentTimeMillis() < end) {
                if (screen().contains(text)) return screen()
                Thread.sleep(100)
            }
            return screen()
        }

        type("tmux -L apptest -f /dev/null new -s app\n")
        var s = waitFor("[app]")
        assertTrue("tmux status bar missing:\n$s", s.contains("[app]"))
        type("echo cols=\$(tput cols) term=\$TERM\r")
        s = waitFor("cols=80")
        assertTrue("inside tmux:\n$s", s.contains("cols=80 term=tmux-256color") || s.contains("cols=80 term=screen"))
        type("\u0002d") // Ctrl-B d: detach
        s = waitFor("[detached")
        assertTrue("detach:\n$s", s.contains("[detached"))
        client.exec("tmux -L apptest kill-server")
        client.disconnect()
        reader.join(2000)
    }

    private class Replies(private val out: OutputStream) : TerminalOutput() {
        override fun write(data: ByteArray, offset: Int, count: Int) {
            out.write(data, offset, count)
            out.flush()
        }

        override fun titleChanged(oldTitle: String?, newTitle: String?) {}
        override fun onCopyTextToClipboard(text: String?) {}
        override fun onPasteTextFromClipboard() {}
        override fun onBell() {}
        override fun onColorsChanged() {}
    }

    /**
     * SshTerminal end to end, as the Shell tab drives it: keys sent while the read loop
     * runs must reach the robot (they used to queue behind it forever), and a resize
     * from the view must reach the PTY.
     */
    @Test
    fun terminalSendsKeysWhileReading() {
        val port = System.getenv("SSH_TEST_PORT")?.toIntOrNull()
        assumeTrue(port != null)
        val ui = Executors.newSingleThreadExecutor()
        val term = SshTerminal(
            SshTarget("127.0.0.1", port!!, System.getProperty("user.name")), null,
            File(System.getenv("SSH_TEST_KEYS")!!), post = { ui.execute(it) },
        )
        fun screen(): String = ui.submit<String> { term.emulator.screen.transcriptText }.get()
        fun waitFor(text: String): Boolean {
            val end = System.currentTimeMillis() + 8000
            while (System.currentTimeMillis() < end) {
                if (screen().contains(text)) return true
                Thread.sleep(100)
            }
            return false
        }
        val end = System.currentTimeMillis() + 8000
        while (term.state != SshTerminal.State.CONNECTED && System.currentTimeMillis() < end) Thread.sleep(50)
        assertEquals(SshTerminal.State.CONNECTED, term.state)

        term.send("echo got-\$((6*7))\r")
        assertTrue("typed command did not run:\n${screen()}", waitFor("got-42"))
        ui.submit { term.resize(100, 30, 1000, 600) }.get()
        term.send("echo cols=\$(tput cols)\r")
        assertTrue("resize did not reach the PTY:\n${screen()}", waitFor("cols=100"))
        term.close()
        ui.shutdown()
        ui.awaitTermination(2, TimeUnit.SECONDS)
    }

    /** The tmux row's "+win" and window keys, and a drag sent as wheel steps into tmux's history. */
    @Test
    fun tmuxButtonsAndWheelScroll() {
        val port = System.getenv("SSH_TEST_PORT")?.toIntOrNull()
        assumeTrue(port != null)
        val ui = Executors.newSingleThreadExecutor()
        val term = SshTerminal(
            SshTarget("127.0.0.1", port!!, System.getProperty("user.name")), null,
            File(System.getenv("SSH_TEST_KEYS")!!), post = { ui.execute(it) },
        )
        fun screen(): String = ui.submit<String> { term.emulator.screen.transcriptText }.get()
        fun waitFor(what: String, test: (String) -> Boolean): String {
            val end = System.currentTimeMillis() + 8000
            while (System.currentTimeMillis() < end) {
                val s = screen()
                if (test(s)) return s
                Thread.sleep(100)
            }
            throw AssertionError("$what:\n${screen()}")
        }
        try {
            val end = System.currentTimeMillis() + 8000
            while (term.state != SshTerminal.State.CONNECTED && System.currentTimeMillis() < end) Thread.sleep(50)
            ui.submit { term.resize(80, 24, 800, 480) }.get()
            term.send("tmux -L apptest -f /dev/null new -A -s robot \\; set -g mouse on\r")
            waitFor("tmux did not start") { it.contains("[robot]") }

            term.send("\u0002c")
            waitFor("+win made no window 1") { it.contains("1:") }
            term.send("\u00020")
            term.send("seq 1 300\r")
            waitFor("seq did not run") { it.contains("300") }
            assertTrue(ui.submit<Boolean> { term.emulator.isMouseTrackingActive }.get())

            ui.submit {
                repeat(20) { term.emulator.sendMouseEvent(TerminalEmulator.MOUSE_WHEELUP_BUTTON, 10, 10, true) }
            }.get()
            waitFor("wheel did not scroll tmux") { Regex("""\[\d+/\d+]""").containsMatchIn(it) }
        } finally {
            ProcessBuilder("tmux", "-L", "apptest", "kill-server").start().waitFor()
            term.close()
            ui.shutdown()
            ui.awaitTermination(2, TimeUnit.SECONDS)
        }
    }
}
