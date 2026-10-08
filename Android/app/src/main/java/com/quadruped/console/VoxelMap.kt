package com.quadruped.console

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.floor

/**
 * Point-LIO's voxel map as the app keeps it: a snapshot replaces everything, deltas append,
 * one point per voxel. Points only ever get appended between snapshots, so the renderer
 * uploads just the tail ([generation] changes on a replace). Thread-safe.
 */
class VoxelMap(val capacity: Int = 300_000, voxel: Double = 0.1) {
    private val keys = LongSet(capacity)
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
        keys.clear()
        size = 0
        overflow = 0
        generation++
        append(points)
    }

    /** Returns how many points were new. */
    @Synchronized
    fun add(points: FloatBuffer): Int {
        val before = size
        append(points)
        return size - before
    }

    @Synchronized
    fun setVoxel(size: Double) {
        if (size == voxel) return
        voxel = size
        clear()
    }

    @Synchronized
    fun clear() {
        keys.clear()
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

    private fun append(points: FloatBuffer) {
        while (points.remaining() >= 3) {
            val x = points.get()
            val y = points.get()
            val z = points.get()
            if (!x.isFinite() || !y.isFinite() || !z.isFinite()) continue
            val k = key(x, y, z)
            if (size == capacity) {
                if (!keys.contains(k)) overflow++
                continue
            }
            if (!keys.add(k)) continue
            xyz[size * 3] = x
            xyz[size * 3 + 1] = y
            xyz[size * 3 + 2] = z
            size++
        }
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

/** Open-addressing set of longs: a few hundred thousand keys without boxing. */
class LongSet(expected: Int) {
    private var table = LongArray(Integer.highestOneBit(maxOf(expected, 8) * 2 - 1) shl 1)
    private var hasZero = false
    var size = 0
        private set

    fun add(k: Long): Boolean {
        if (k == 0L) {
            if (hasZero) return false
            hasZero = true
            size++
            return true
        }
        if ((size + 1) * 2 > table.size) grow()
        return insert(table, k).also { if (it) size++ }
    }

    fun contains(k: Long): Boolean {
        if (k == 0L) return hasZero
        val mask = table.size - 1
        var i = mix(k) and mask
        while (true) {
            val v = table[i]
            if (v == 0L) return false
            if (v == k) return true
            i = (i + 1) and mask
        }
    }

    fun clear() {
        table.fill(0L)
        hasZero = false
        size = 0
    }

    private fun insert(t: LongArray, k: Long): Boolean {
        val mask = t.size - 1
        var i = mix(k) and mask
        while (true) {
            val v = t[i]
            if (v == 0L) {
                t[i] = k
                return true
            }
            if (v == k) return false
            i = (i + 1) and mask
        }
    }

    private fun grow() {
        val next = LongArray(table.size * 2)
        for (v in table) if (v != 0L) insert(next, v)
        table = next
    }

    private fun mix(k: Long): Int {
        var h = k * -0x61c8864680b583ebL
        h = h xor (h ushr 32)
        return h.toInt()
    }
}
