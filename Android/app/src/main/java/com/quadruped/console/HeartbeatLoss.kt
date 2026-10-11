package com.quadruped.console

/**
 * Share of our heartbeats the robot lost, per second over the last [seconds], from the running totals
 * (received, missed) that /robot_state's "heartbeat" carries. A second with no report is NaN.
 */
class HeartbeatLoss(private val seconds: Int = 60) {
    private val buckets = ArrayDeque<LongArray>()  // [received, lost], or empty for no report
    private var base: LongArray? = null
    private var last: LongArray? = null
    private var bucketStartMs = -1L

    fun add(nowMs: Long, received: Long, missed: Long) {
        roll(nowMs)
        val l = last
        // A restarted robot starts its totals from 0.
        if (l != null && (received < l[0] || missed < l[1])) base = null
        last = longArrayOf(received, missed)
        if (base == null) {
            base = last
            if (bucketStartMs < 0) bucketStartMs = nowMs
        }
    }

    /** Closes every second that has ended; seconds without a report stay empty. */
    fun roll(nowMs: Long) {
        if (bucketStartMs < 0) return
        while (nowMs - bucketStartMs >= 1000) {
            val b = base
            val l = last
            if (b != null && l != null && (l[0] > b[0] || l[1] > b[1])) push(longArrayOf(l[0] - b[0], l[1] - b[1]))
            else push(LongArray(0))
            base = l
            bucketStartMs += 1000
        }
    }

    private fun push(b: LongArray) {
        buckets.addLast(b)
        while (buckets.size > seconds) buckets.removeFirst()
    }

    /** Oldest first; NaN for a second with no report. */
    fun history(): DoubleArray = DoubleArray(buckets.size) {
        val b = buckets[it]
        if (b.isEmpty()) Double.NaN else b[1].toDouble() / (b[0] + b[1])
    }

    /** Lost / sent over the whole window, in %, or null before the first full second. */
    fun percent(): Double? {
        val full = buckets.filter { it.isNotEmpty() }
        val sent = full.sumOf { it[0] + it[1] }
        return if (sent == 0L) null else 100.0 * full.sumOf { it[1] } / sent
    }
}
