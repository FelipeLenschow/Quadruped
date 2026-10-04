package com.quadruped.console.ui

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.graphics.Typeface
import android.text.InputType
import android.util.TypedValue
import android.view.GestureDetector
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.inputmethod.BaseInputConnection
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.view.inputmethod.InputMethodManager
import com.quadruped.console.ssh.SshTerminal
import com.termux.terminal.KeyHandler
import com.termux.view.TerminalRenderer
import kotlin.math.roundToInt

/**
 * Draws an [SshTerminal] with Termux's renderer and turns the soft keyboard into bytes.
 * Tap: keyboard. Drag: scrollback. Pinch: text size.
 */
@SuppressLint("ViewConstructor")
class TerminalCanvasView(context: Context) : View(context) {
    var terminal: SshTerminal? = null
        set(value) {
            field?.onUpdate = null
            field = value
            value?.onUpdate = { invalidate() }
            topRow = 0
            fitToSize()
            invalidate()
        }

    /** Sticky Ctrl from the extra keys row: applies to the next key, then clears. */
    var ctrlArmed = false
    var onCtrlConsumed: (() -> Unit)? = null

    private var textSizePx = sp(12f)
    private var renderer = TerminalRenderer(textSizePx, Typeface.MONOSPACE)
    private var topRow = 0
    private var scrollRemainder = 0f

    init {
        isFocusable = true
        isFocusableInTouchMode = true
    }

    private fun sp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, v, resources.displayMetrics).roundToInt()

    private fun fitToSize() {
        val t = terminal ?: return
        if (width == 0 || height == 0) return
        val cols = (width / renderer.fontWidth).toInt()
        val rows = height / renderer.fontLineSpacing
        t.resize(cols, rows, width, height)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) = fitToSize()

    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(0xFF0B0D12.toInt())
        val emu = terminal?.emulator ?: return
        topRow = topRow.coerceIn(-emu.screen.activeTranscriptRows, 0)
        renderer.render(emu, canvas, topRow, -1, -1, -1, -1)
    }

    // ---------------------------------------------------------------- touch

    private val gestures = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onSingleTapUp(e: MotionEvent): Boolean {
            showKeyboard()
            return true
        }

        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean {
            scrollRemainder += dy
            val lines = (scrollRemainder / renderer.fontLineSpacing).toInt()
            if (lines != 0) {
                scrollRemainder -= lines * renderer.fontLineSpacing
                topRow += lines
                invalidate()
            }
            return true
        }
    })

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val size = (textSizePx * d.scaleFactor).roundToInt().coerceIn(sp(7f), sp(28f))
            if (size != textSizePx) {
                textSizePx = size
                renderer = TerminalRenderer(textSizePx, Typeface.MONOSPACE)
                fitToSize()
                invalidate()
            }
            return true
        }
    })

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaler.onTouchEvent(event)
        if (!scaler.isInProgress) gestures.onTouchEvent(event)
        return true
    }

    fun showKeyboard() {
        requestFocus()
        context.getSystemService(InputMethodManager::class.java).showSoftInput(this, InputMethodManager.SHOW_IMPLICIT)
    }

    // ---------------------------------------------------------------- keys

    /** Text from the keyboard or a shortcut; Ctrl and the scroll position apply. */
    fun sendText(text: String) {
        val t = terminal ?: return
        topRow = 0
        val out = if (ctrlArmed && text.length == 1) {
            ctrlArmed = false
            onCtrlConsumed?.invoke()
            controlOf(text[0])
        } else {
            text.replace('\n', '\r')
        }
        t.send(out)
    }

    /** Special keys (arrows, Esc, Tab...) in the escape form the remote app asked for. */
    fun sendKey(keyCode: Int, ctrl: Boolean = false): Boolean {
        val t = terminal ?: return false
        val emu = t.emulator
        val mods = if (ctrl) KeyHandler.KEYMOD_CTRL else 0
        val code = KeyHandler.getCode(keyCode, mods, emu.isCursorKeysApplicationMode, emu.isKeypadApplicationMode)
            ?: return false
        topRow = 0
        t.send(code)
        return true
    }

    private fun controlOf(c: Char): String = when (c) {
        in 'a'..'z' -> (c - 'a' + 1).toChar().toString()
        in 'A'..'Z' -> (c - 'A' + 1).toChar().toString()
        ' ', '2', '@' -> "\u0000"
        '[', '3' -> "\u001b"
        '\\', '4' -> "\u001c"
        ']', '5' -> "\u001d"
        '^', '6' -> "\u001e"
        '_', '7', '/' -> "\u001f"
        '?', '8' -> "\u007f"
        else -> c.toString()
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (terminal == null) return super.onKeyDown(keyCode, event)
        if (keyCode == KeyEvent.KEYCODE_BACK) return super.onKeyDown(keyCode, event)
        val ctrl = event.isCtrlPressed || ctrlArmed
        if (sendKey(keyCode, ctrl)) {
            if (ctrlArmed) {
                ctrlArmed = false
                onCtrlConsumed?.invoke()
            }
            return true
        }
        val ch = event.getUnicodeChar(event.metaState and KeyEvent.META_CTRL_MASK.inv())
        if (ch == 0) return super.onKeyDown(keyCode, event)
        if (event.isCtrlPressed) ctrlArmed = true
        sendText(String(Character.toChars(ch)))
        return true
    }

    override fun onCheckIsTextEditor() = true

    override fun onCreateInputConnection(outAttrs: EditorInfo): InputConnection {
        // Visible-password: no autocorrect, no suggestions, no composing, like Termux.
        outAttrs.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD or
            InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        outAttrs.imeOptions = EditorInfo.IME_FLAG_NO_FULLSCREEN or EditorInfo.IME_FLAG_NO_EXTRACT_UI
        return object : BaseInputConnection(this, true) {
            override fun commitText(text: CharSequence, newCursorPosition: Int): Boolean {
                super.commitText(text, newCursorPosition)
                flushEditable()
                return true
            }

            override fun finishComposingText(): Boolean {
                super.finishComposingText()
                flushEditable()
                return true
            }

            override fun deleteSurroundingText(beforeLength: Int, afterLength: Int): Boolean {
                repeat(maxOf(1, beforeLength)) { sendText("\u007f") }
                return true
            }

            override fun sendKeyEvent(event: KeyEvent): Boolean {
                if (event.action == KeyEvent.ACTION_DOWN) onKeyDown(event.keyCode, event)
                return true
            }

            private fun flushEditable() {
                val e = editable ?: return
                val text = e.toString()
                e.clear()
                if (text.isEmpty()) return
                // One character at a time so a sticky Ctrl applies to the first only.
                text.codePoints().forEach { sendText(String(Character.toChars(it))) }
            }
        }
    }
}
