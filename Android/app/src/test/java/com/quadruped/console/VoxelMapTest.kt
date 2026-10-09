package com.quadruped.console

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class VoxelMapTest {
    private fun pts(vararg v: Float): FloatBuffer = FloatBuffer.wrap(v)
    private fun out(n: Int): FloatBuffer = FloatBuffer.allocate(n * 3)

    @Test
    fun deltasDedupeByVoxel() {
        val m = VoxelMap()
        assertEquals(2, m.add(pts(0.05f, 0.05f, 0.05f, 1.05f, 2.05f, -0.25f)))
        // Same voxels again, one nudged within its cell, plus the one above.
        assertEquals(1, m.add(pts(0.09f, 0.01f, 0.05f, 1.05f, 2.05f, -0.25f, 1.05f, 2.05f, -0.15f)))
        assertEquals(3, m.size)
    }

    @Test
    fun neighbouringCentresNeverShareAKey() {
        // lio_map_stream's centres, (k + 0.5) * 0.1, across a range where float error bites.
        val m = VoxelMap()
        val centres = (-200..200).map { ((it + 0.5) * 0.1).toFloat() }
        assertEquals(centres.size, centres.map { m.key(it, it, it) }.toSet().size)
    }

    @Test
    fun snapshotReplacesEvenWhenSmaller() {
        val m = VoxelMap()
        m.add(pts(0f, 0f, 0f, 1f, 0f, 0f, 2f, 0f, 0f))
        val gen = m.generation
        m.replace(pts(5f, 5f, 0f))
        assertEquals(1, m.size)
        assertTrue(m.generation != gen)
        // The old voxels are gone, so they count as new again.
        assertEquals(1, m.add(pts(0f, 0f, 0f)))
    }

    @Test
    fun capacityStopsAndCountsOverflow() {
        val m = VoxelMap(capacity = 2)
        m.add(pts(0f, 0f, 0f, 1f, 0f, 0f, 2f, 0f, 0f, 3f, 0f, 0f, 1f, 0f, 0f))
        assertEquals(2, m.size)
        assertEquals(2L, m.overflow)
    }

    @Test
    fun nonFinitePointsAreSkipped() {
        val m = VoxelMap()
        m.add(pts(0f, Float.NaN, 0f, 0f, Float.POSITIVE_INFINITY, 0f, Float.POSITIVE_INFINITY, 0f, 0f, 1f, 1f, 1f))
        assertEquals(1, m.size)
    }

    private val removeMark = floatArrayOf(Float.NaN, 0f, 0f)
    private val addMark = floatArrayOf(Float.NaN, 1f, 0f)
    private fun delta(removed: FloatArray, added: FloatArray): FloatBuffer =
        FloatBuffer.wrap(removeMark + removed + addMark + added)

    @Test
    fun deltasRemoveThenAdd() {
        val m = VoxelMap()
        m.add(pts(0.05f, 0.05f, 0.05f, 1.05f, 0.05f, 0.05f, 2.05f, 0.05f, 0.05f))
        val gen = m.generation
        assertEquals(1, m.add(delta(floatArrayOf(0.05f, 0.05f, 0.05f), floatArrayOf(3.05f, 0.05f, 0.05f))))
        assertEquals(3, m.size)
        assertTrue(m.generation != gen)
        // The removed voxel is new again; the others, including the one moved into its slot, are still known.
        assertEquals(1, m.add(pts(0.05f, 0.05f, 0.05f, 1.05f, 0.05f, 0.05f, 2.05f, 0.05f, 0.05f, 3.05f, 0.05f, 0.05f)))
        assertEquals(4, m.size)
    }

    @Test
    fun removingWhatIsNotThereChangesNothing() {
        val m = VoxelMap()
        m.add(pts(0.05f, 0.05f, 0.05f))
        val gen = m.generation
        m.add(delta(floatArrayOf(5.05f, 5.05f, 5.05f), floatArrayOf()))
        assertEquals(1, m.size)
        assertEquals(gen, m.generation)
    }

    @Test
    fun removalsKeepTheRestIntact() {
        val m = VoxelMap()
        val all = (0 until 50).flatMap { listOf((it + 0.5f) * 0.1f, 0.05f, 0.05f) }.toFloatArray()
        m.add(FloatBuffer.wrap(all))
        val odd = (1 until 50 step 2).flatMap { listOf((it + 0.5f) * 0.1f, 0.05f, 0.05f) }.toFloatArray()
        m.add(delta(odd, floatArrayOf()))
        assertEquals(25, m.size)
        val buf = out(m.capacity)
        m.copyNew(-1, 0, buf)
        val xs = List(25) { buf.get(it * 3) }.map { Math.round(it * 10 - 0.5f) }.sorted()
        assertEquals((0 until 50 step 2).toList(), xs)
    }

    @Test
    fun copyNewGivesTheTailThenEverythingAfterReplace() {
        val m = VoxelMap()
        val buf = out(m.capacity)
        m.add(pts(0f, 0f, 0f, 1f, 0f, 0f))
        var t = m.copyNew(-1, 0, buf)
        assertEquals(0, t.from)
        assertEquals(2, t.to)
        assertEquals(6, buf.remaining())

        m.add(pts(2f, 0f, 0f))
        t = m.copyNew(t.generation, t.to, buf)
        assertEquals(2, t.from)
        assertEquals(3, t.to)
        assertEquals(2f, buf.get(0))

        m.replace(pts(9f, 9f, 9f))
        t = m.copyNew(t.generation, t.to, buf)
        assertEquals(0, t.from)
        assertEquals(1, t.to)
        assertEquals(9f, buf.get(0))
    }

    @Test
    fun floatsReadsTheLinkLittleEndianBytes() {
        val b = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN).putFloat(1.5f).putFloat(-2f).putFloat(0.25f)
        val m = VoxelMap()
        m.replace(VoxelMap.floats(b.array()))
        val buf = out(1)
        m.copyNew(-1, 0, buf)
        assertEquals(listOf(1.5f, -2f, 0.25f), List(3) { buf.get(it) })
    }

    @Test
    fun longIntMapGrowsRemovesAndHandlesZeroAndNegatives() {
        val s = LongIntMap(4)
        s.put(0L, 7)
        assertEquals(7, s.get(0L))
        for (i in 1..10_000L) s.put(-i * 7919, i.toInt())
        assertEquals(10_001, s.size)
        for (i in 1..10_000L step 3) assertEquals(i.toInt(), s.remove(-i * 7919))
        for (i in 1..10_000L) assertEquals(if ((i - 1) % 3 == 0L) -1 else i.toInt(), s.get(-i * 7919))
        assertEquals(-1, s.get(5L))
        assertEquals(7, s.remove(0L))
        assertEquals(-1, s.get(0L))
        assertEquals(10_000 - 3334, s.size)
        s.clear()
        assertEquals(-1, s.get(-2 * 7919L))
        assertEquals(0, s.size)
    }

    @Test
    fun keysAreDistinctAcrossAxesAndSigns() {
        val m = VoxelMap()
        val keys = listOf(
            m.key(0.15f, 0.05f, 0.05f), m.key(0.05f, 0.15f, 0.05f), m.key(0.05f, 0.05f, 0.15f),
            m.key(-0.05f, 0.05f, 0.05f), m.key(0.05f, -0.05f, 0.05f), m.key(0.05f, 0.05f, -0.05f),
        )
        assertEquals(keys.size, keys.toSet().size)
    }
}
