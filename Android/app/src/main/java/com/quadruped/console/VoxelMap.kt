package com.quadruped.console

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.floor

/**
 * Point-LIO's voxel map as the app keeps it: a snapshot replaces everything, deltas add and
 * remove, one point per voxel. A delta from lio_map_stream is a remove mark, the voxels to
 * remove, an add mark, the voxels to add; the marks have x = NaN and y = 0 (remove) or 1 (add).
 * Additions append, so the renderer uploads just the tail; [generation] changes on a replace or
 * a removal, which moves points. Thread-safe.
 */
class VoxelMap(val capacity: Int = 300_000, voxel: Double = 0.1) {
    private val index = LongIntMap(capacity)
    private val xyz = FloatArray(capacity * 3)

    /** Voxel edge in m, as /lio/map_voxels/size announces it. Keys depend on it, so a change clears the map. */
    @Volatile var voxel = voxel
        private set

    @Volatile var size = 0
        private set
    @Volatile var generation = 0
        private set
    /** Points that did not fit; the map is full. */
    @Volatile var overflow = 0L
        private set

    @Synchronized
    fun replace(points: FloatBuffer) {
        index.clear()
        size = 0
        overflow = 0
        generation++
        apply(points)
    }

    /** Returns how many points were new. */
    @Synchronized
    fun add(points: FloatBuffer): Int = apply(points)

    @Synchronized
    fun setVoxel(size: Double) {
        if (size == voxel) return
        voxel = size
        clear()
    }

    @Synchronized
    fun clear() {
        index.clear()
        size = 0
        overflow = 0
        generation++
    }

    class Tail(val generation: Int, val from: Int, val to: Int)

    /**
     * What a reader holding [have] points of [generation] is missing, copied into [out]:
     * the new tail, or everything after a replace. One locked step, so a snapshot
     * landing in between cannot mix two maps.
     */
    @Synchronized
    fun copyNew(generation: Int, have: Int, out: FloatBuffer): Tail {
        val from = if (generation == this.generation) minOf(have, size) else 0
        out.clear()
        out.put(xyz, from * 3, (size - from) * 3)
        out.flip()
        return Tail(this.generation, from, size)
    }

    private fun apply(points: FloatBuffer): Int {
        var adding = true
        var added = 0
        var removed = false
        while (points.remaining() >= 3) {
            val x = points.get()
            val y = points.get()
            val z = points.get()
            if (x.isNaN()) {
                adding = y > 0.5f
                continue
            }
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) continue
            val k = key(x, y, z)
            if (!adding) {
                removed = remove(k) || removed
                continue
            }
            if (index.get(k) >= 0) continue
            if (size == capacity) {
                overflow++
                continue
            }
            index.put(k, size)
            xyz[size * 3] = x
            xyz[size * 3 + 1] = y
            xyz[size * 3 + 2] = z
            size++
            added++
        }
        if (removed) generation++
        return added
    }

    /** The last point moves into the hole. */
    private fun remove(k: Long): Boolean {
        val i = index.remove(k)
        if (i < 0) return false
        val last = size - 1
        if (i != last) {
            System.arraycopy(xyz, last * 3, xyz, i * 3, 3)
            index.put(key(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]), i)
        }
        size--
        return true
    }

    /**
     * floor(x / voxel) per axis, 21 bits each, as lio_map_stream keys them. Its points are
     * voxel centres, (k + 0.5) * voxel, mid-cell for floor; round() would sit on the .5.
     */
    fun key(x: Float, y: Float, z: Float): Long {
        fun k(v: Float) = floor(v / voxel).toLong() and 0x1FFFFF
        return (k(x) shl 42) or (k(y) shl 21) or k(z)
    }

    companion object {
        /** The link's packed x,y,z float32 little-endian bytes as floats. */
        fun floats(bytes: ByteArray): FloatBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    }
}

/** Open-addressing map of long keys to non-negative ints, with deletion: a few hundred thousand keys without boxing. */
class LongIntMap(expected: Int) {
    private var keys = LongArray(Integer.highestOneBit(maxOf(expected, 8) * 2 - 1) shl 1)
    private var vals = IntArray(keys.size)
    private var zeroValue = -1
    var size = 0
        private set

    /** The value for [k], or -1. */
    fun get(k: Long): Int {
        if (k == 0L) return zeroValue
        val mask = keys.size - 1
        var i = mix(k) and mask
        while (true) {
            val v = keys[i]
            if (v == 0L) return -1
            if (v == k) return vals[i]
            i = (i + 1) and mask
        }
    }

    fun put(k: Long, value: Int) {
        if (k == 0L) {
            if (zeroValue < 0) size++
            zeroValue = value
            return
        }
        if ((size + 1) * 2 > keys.size) grow()
        if (insert(keys, vals, k, value)) size++
    }

    /** Removes [k]; returns its value, or -1. */
    fun remove(k: Long): Int {
        if (k == 0L) {
            val v = zeroValue
            if (v >= 0) {
                zeroValue = -1
                size--
            }
            return v
        }
        val mask = keys.size - 1
        var i = mix(k) and mask
        while (keys[i] != k) {
            if (keys[i] == 0L) return -1
            i = (i + 1) and mask
        }
        val out = vals[i]
        // Backward shift: later keys of the same probe run that may move into the hole do.
        var hole = i
        var j = i
        while (true) {
            j = (j + 1) and mask
            val kj = keys[j]
            if (kj == 0L) break
            val home = mix(kj) and mask
            val stays = if (hole <= j) home in hole + 1..j else home > hole || home <= j
            if (!stays) {
                keys[hole] = kj
                vals[hole] = vals[j]
                hole = j
            }
        }
        keys[hole] = 0L
        size--
        return out
    }

    fun clear() {
        keys.fill(0L)
        zeroValue = -1
        size = 0
    }

    private fun insert(t: LongArray, v: IntArray, k: Long, value: Int): Boolean {
        val mask = t.size - 1
        var i = mix(k) and mask
        while (true) {
            val e = t[i]
            if (e == 0L) {
                t[i] = k
                v[i] = value
                return true
            }
            if (e == k) {
                v[i] = value
                return false
            }
            i = (i + 1) and mask
        }
    }

    private fun grow() {
        val nk = LongArray(keys.size * 2)
        val nv = IntArray(nk.size)
        for (i in keys.indices) if (keys[i] != 0L) insert(nk, nv, keys[i], vals[i])
        keys = nk
        vals = nv
    }

    private fun mix(k: Long): Int {
        var h = k * -0x61c8864680b583ebL
        h = h xor (h ushr 32)
        return h.toInt()
    }
}
