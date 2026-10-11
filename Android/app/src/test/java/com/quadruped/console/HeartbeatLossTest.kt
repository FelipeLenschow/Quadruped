package com.quadruped.console

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HeartbeatLossTest {
    @Test
    fun perSecondShareAndWindowPercent() {
        val h = HeartbeatLoss()
        assertNull(h.percent())
        var t = 0L
        var got = 0L
        var lost = 0L
        // 10 Hz reports; second 2 loses 2 of 10 heartbeats, second 4 has no reports at all.
        for (sec in 0 until 6) {
            for (i in 0 until 10) {
                if (sec == 2 && i < 2) lost++ else got++
                if (sec != 4) h.add(t, got, lost)
                t += 100
            }
        }
        h.roll(t)
        val hist = h.history()
        assertEquals(6, hist.size)
        assertEquals(0.2, hist[2], 1e-9)
        assertEquals(0.0, hist[1], 1e-9)
        // Second 4's heartbeats are only reported in second 5, so 4 is empty and 5 counts both.
        assertTrue(hist[4].isNaN())
        assertEquals(0.0, hist[5], 1e-9)
        assertEquals(100.0 * 2 / 59, h.percent()!!, 1e-9)
    }

    @Test
    fun robotRestartDoesNotCountNegative() {
        val h = HeartbeatLoss()
        h.add(0, 500, 7)
        h.add(500, 505, 7)
        h.add(1000, 3, 0)
        h.add(2000, 13, 1)
        h.roll(3000)
        val hist = h.history()
        assertTrue(hist.all { it.isNaN() || it in 0.0..1.0 })
        assertEquals(1.0 / 11, hist.last(), 1e-9)
    }
}
