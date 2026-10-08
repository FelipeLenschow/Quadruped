package com.quadruped.console

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
        m.add(pts(Float.NaN, 0f, 0f, 0f, Float.POSITIVE_INFINITY, 0f, 1f, 1f, 1f))
        assertEquals(1, m.size)
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
    fun longSetGrowsAndHandlesZeroAndNegatives() {
        val s = LongSet(4)
        assertTrue(s.add(0L))
        assertFalse(s.add(0L))
        for (i in 1..10_000L) assertTrue(s.add(-i * 7919))
        for (i in 1..10_000L) assertTrue(s.contains(-i * 7919))
        assertFalse(s.contains(5L))
        assertEquals(10_001, s.size)
        s.clear()
        assertFalse(s.contains(0L))
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
