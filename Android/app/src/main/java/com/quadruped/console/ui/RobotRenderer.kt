package com.quadruped.console.ui

import android.content.res.AssetManager
import android.opengl.GLES30.*
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.util.Log
import com.quadruped.console.Pose3
import com.quadruped.console.Urdf
import com.quadruped.console.Vec3
import com.quadruped.console.VoxelMap
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLDisplay
import javax.microedition.khronos.opengles.GL10
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.sin

/**
 * What to draw: every visual link's pose in the world frame, plus the base's trail.
 * Without [grid] (a 3D map has its own floor) the trail follows the base's height.
 */
class RobotScene(
    val base: Pose3, val links: Map<String, Pose3>, val trail: List<Vec3>,
    val grid: Boolean = true, val cloud: VoxelMap? = null,
)

/** Orbit camera around [focus], or the base when it is null; written from the UI thread. */
class OrbitCamera {
    @Volatile var yaw = -2.3f
    @Volatile var pitch = 0.4f
    @Volatile var dist = 1.4f
    @Volatile var focus: Vec3? = null
}

/** Draws the URDF's meshes (exported by tools/export_urdf.py) with OpenGL ES 3. */
class RobotRenderer(
    private val assets: AssetManager,
    private val urdf: Urdf,
    val camera: OrbitCamera,
) : GLSurfaceView.Renderer {
    @Volatile var scene: RobotScene? = null

    private class Part(val vao: Int, val count: Int, val color: FloatArray)

    private val meshes = HashMap<String, List<Part>>()
    private var litProgram = 0
    private var lineProgram = 0
    private var boxProgram = 0
    private var boxVao = 0
    private var centreVbo = 0
    private var boxCapacity = 0
    private var uploaded = 0
    private var uploadedGeneration = -1
    private var centreData: FloatBuffer? = null
    private var aspect = 1f
    private var lineVao = 0
    private var lineVbo = 0
    private val proj = FloatArray(16)
    private val view = FloatArray(16)
    private val viewProj = FloatArray(16)
    private val model = FloatArray(16)
    private val mvp = FloatArray(16)
    private val identity = FloatArray(16).also { Matrix.setIdentityM(it, 0) }
    private val lineData: FloatBuffer = ByteBuffer.allocateDirect(4 * 3 * 8192).order(ByteOrder.nativeOrder()).asFloatBuffer()

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        glClearColor(0x0B / 255f, 0x0D / 255f, 0x12 / 255f, 1f)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        litProgram = program(LIT_VS, LIT_FS)
        lineProgram = program(LINE_VS, LINE_FS)
        boxProgram = program(BOX_VS, BOX_FS)
        // A new GL context: the old buffers are gone.
        boxVao = 0
        uploadedGeneration = -1
        meshes.clear()
        urdf.visuals.values.flatten().map { it.mesh }.toSet().forEach { name ->
            try {
                meshes[name] = loadMesh(name)
            } catch (e: Exception) {
                Log.e("RobotRenderer", "mesh $name", e)
            }
        }
        val ids = IntArray(2)
        glGenVertexArrays(1, ids, 0)
        glGenBuffers(1, ids, 1)
        lineVao = ids[0]
        lineVbo = ids[1]
        glBindVertexArray(lineVao)
        glBindBuffer(GL_ARRAY_BUFFER, lineVbo)
        glBufferData(GL_ARRAY_BUFFER, lineData.capacity() * 4, null, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 12, 0)
        glBindVertexArray(0)
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        glViewport(0, 0, width, height)
        aspect = width.toFloat() / height
    }

    override fun onDrawFrame(gl: GL10?) {
        glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        val s = scene ?: return
        val c = camera
        val target = c.focus ?: s.base.t
        // A near plane this far out keeps depth precision for the overlapping shell parts.
        Matrix.perspectiveM(proj, 0, 38f, aspect, 0.1f, maxOf(40f, c.dist * 4))
        val eye = target + Vec3(cos(c.pitch) * cos(c.yaw).toDouble(), cos(c.pitch) * sin(c.yaw).toDouble(), sin(c.pitch).toDouble()) * c.dist.toDouble()
        Matrix.setLookAtM(
            view, 0, eye.x.toFloat(), eye.y.toFloat(), eye.z.toFloat(),
            target.x.toFloat(), target.y.toFloat(), target.z.toFloat(), 0f, 0f, 1f,
        )
        Matrix.multiplyMM(viewProj, 0, proj, 0, view, 0)

        if (s.grid) drawGrid(target)
        s.cloud?.let(::drawCloud)
        drawTrail(s.trail, flat = s.grid)

        glUseProgram(litProgram)
        glUniform3f(glGetUniformLocation(litProgram, "uLight"), 0.36f, 0.48f, 0.80f)
        glUniform3f(glGetUniformLocation(litProgram, "uEye"), eye.x.toFloat(), eye.y.toFloat(), eye.z.toFloat())
        val uMvp = glGetUniformLocation(litProgram, "uMVP")
        val uModel = glGetUniformLocation(litProgram, "uModel")
        val uColor = glGetUniformLocation(litProgram, "uColor")
        for ((link, visuals) in urdf.visuals) {
            val pose = s.links[link] ?: continue
            for (v in visuals) {
                val parts = meshes[v.mesh] ?: continue
                toMatrix(pose * v.origin, model)
                Matrix.multiplyMM(mvp, 0, viewProj, 0, model, 0)
                glUniformMatrix4fv(uMvp, 1, false, mvp, 0)
                glUniformMatrix4fv(uModel, 1, false, model, 0)
                for (p in parts) {
                    glUniform4fv(uColor, 1, p.color, 0)
                    glBindVertexArray(p.vao)
                    glDrawElements(GL_TRIANGLES, p.count, GL_UNSIGNED_INT, 0)
                }
            }
        }
        glBindVertexArray(0)
    }

    private fun drawGrid(around: Vec3) {
        val step = 0.5
        val n = 10
        val ox = floor(around.x / step) * step
        val oy = floor(around.y / step) * step
        lineData.clear()
        for (i in -n..n) {
            val x = ox + i * step
            val y = oy + i * step
            lineData.put(floatArrayOf(x.toFloat(), (oy - n * step).toFloat(), 0f, x.toFloat(), (oy + n * step).toFloat(), 0f))
            lineData.put(floatArrayOf((ox - n * step).toFloat(), y.toFloat(), 0f, (ox + n * step).toFloat(), y.toFloat(), 0f))
        }
        drawLines(GL_LINES, floatArrayOf(0x2A / 255f, 0x31 / 255f, 0x40 / 255f, 1f))
    }

    /**
     * One cube per voxel, instanced: a unit cube's 24 vertices, and the voxel centres as a
     * per-instance attribute. Appends only what is new since the last frame; a new snapshot
     * re-uploads everything.
     */
    private fun drawCloud(map: VoxelMap) {
        if (boxVao == 0 || boxCapacity != map.capacity) {
            val ids = IntArray(4)
            glGenVertexArrays(1, ids, 0)
            glGenBuffers(3, ids, 1)
            boxVao = ids[0]
            centreVbo = ids[3]
            boxCapacity = map.capacity
            centreData = ByteBuffer.allocateDirect(4 * 3 * map.capacity).order(ByteOrder.nativeOrder()).asFloatBuffer()
            glBindVertexArray(boxVao)
            glBindBuffer(GL_ARRAY_BUFFER, ids[1])
            glBufferData(GL_ARRAY_BUFFER, CUBE.size * 4, floatBuffer(CUBE), GL_STATIC_DRAW)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, false, 24, 0)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_FLOAT, false, 24, 12)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ids[2])
            val idx = ByteBuffer.allocateDirect(CUBE_INDICES.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            idx.put(CUBE_INDICES).flip()
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, CUBE_INDICES.size * 2, idx, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, centreVbo)
            glBufferData(GL_ARRAY_BUFFER, 4 * 3 * map.capacity, null, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, false, 12, 0)
            glVertexAttribDivisor(1, 1)
            glBindVertexArray(0)
            uploadedGeneration = -1
        }
        if (map.generation != uploadedGeneration || map.size != uploaded) {
            val data = centreData!!
            val t = map.copyNew(uploadedGeneration, uploaded, data)
            glBindBuffer(GL_ARRAY_BUFFER, centreVbo)
            if (t.to > t.from) glBufferSubData(GL_ARRAY_BUFFER, t.from * 12, (t.to - t.from) * 12, data)
            uploadedGeneration = t.generation
            uploaded = t.to
        }
        if (uploaded == 0) return
        glUseProgram(boxProgram)
        glUniformMatrix4fv(glGetUniformLocation(boxProgram, "uMVP"), 1, false, viewProj, 0)
        glUniform1f(glGetUniformLocation(boxProgram, "uSize"), map.voxel.toFloat())
        glUniform3f(glGetUniformLocation(boxProgram, "uLight"), 0.36f, 0.48f, 0.80f)
        glBindVertexArray(boxVao)
        glDrawElementsInstanced(GL_TRIANGLES, CUBE_INDICES.size, GL_UNSIGNED_SHORT, 0, uploaded)
        glBindVertexArray(0)
    }

    private fun floatBuffer(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().put(a).also { it.flip() }

    private fun drawTrail(trail: List<Vec3>, flat: Boolean) {
        if (trail.size < 2) return
        lineData.clear()
        for (p in trail.takeLast(lineData.capacity() / 3)) {
            lineData.put(floatArrayOf(p.x.toFloat(), p.y.toFloat(), if (flat) 0.002f else p.z.toFloat()))
        }
        drawLines(GL_LINE_STRIP, floatArrayOf(0x3D / 255f, 0xD6 / 255f, 0xC6 / 255f, 0.8f))
    }

    private fun drawLines(mode: Int, color: FloatArray) {
        val count = lineData.position() / 3
        lineData.flip()
        glUseProgram(lineProgram)
        glUniformMatrix4fv(glGetUniformLocation(lineProgram, "uMVP"), 1, false, viewProj, 0)
        glUniform4fv(glGetUniformLocation(lineProgram, "uColor"), 1, color, 0)
        glBindVertexArray(lineVao)
        glBindBuffer(GL_ARRAY_BUFFER, lineVbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, count * 12, lineData)
        glLineWidth(2f)
        glDrawArrays(mode, 0, count)
        glBindVertexArray(0)
    }

    private fun loadMesh(name: String): List<Part> {
        val bytes = assets.open("go2/$name.mesh").use { it.readBytes() }
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { buf.get(it) }
        require(String(magic) == "QMSH") { "bad mesh $name" }
        return List(buf.int) {
            val color = FloatArray(4) { buf.float }
            val nv = buf.int
            val ni = buf.int
            val ids = IntArray(3)
            glGenVertexArrays(1, ids, 0)
            glGenBuffers(2, ids, 1)
            glBindVertexArray(ids[0])
            val vData = buf.slice().order(ByteOrder.LITTLE_ENDIAN).limit(nv * 24) as ByteBuffer
            glBindBuffer(GL_ARRAY_BUFFER, ids[1])
            glBufferData(GL_ARRAY_BUFFER, nv * 24, direct(vData), GL_STATIC_DRAW)
            buf.position(buf.position() + nv * 24)
            val iData = buf.slice().order(ByteOrder.LITTLE_ENDIAN).limit(ni * 4) as ByteBuffer
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ids[2])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, ni * 4, direct(iData), GL_STATIC_DRAW)
            buf.position(buf.position() + ni * 4)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, false, 24, 0)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, false, 24, 12)
            glBindVertexArray(0)
            Part(ids[0], ni, color)
        }
    }

    private fun direct(b: ByteBuffer): ByteBuffer =
        ByteBuffer.allocateDirect(b.remaining()).order(ByteOrder.nativeOrder()).put(b).also { it.flip() }

    private fun program(vs: String, fs: String): Int {
        fun shader(type: Int, src: String) = glCreateShader(type).also {
            glShaderSource(it, src)
            glCompileShader(it)
            val ok = IntArray(1)
            glGetShaderiv(it, GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) Log.e("RobotRenderer", glGetShaderInfoLog(it))
        }
        return glCreateProgram().also {
            glAttachShader(it, shader(GL_VERTEX_SHADER, vs))
            glAttachShader(it, shader(GL_FRAGMENT_SHADER, fs))
            glLinkProgram(it)
        }
    }

    private fun toMatrix(p: Pose3, m: FloatArray) {
        val (x, y, z, w) = p.q
        m[0] = (1 - 2 * (y * y + z * z)).toFloat(); m[1] = (2 * (x * y + z * w)).toFloat(); m[2] = (2 * (x * z - y * w)).toFloat(); m[3] = 0f
        m[4] = (2 * (x * y - z * w)).toFloat(); m[5] = (1 - 2 * (x * x + z * z)).toFloat(); m[6] = (2 * (y * z + x * w)).toFloat(); m[7] = 0f
        m[8] = (2 * (x * z + y * w)).toFloat(); m[9] = (2 * (y * z - x * w)).toFloat(); m[10] = (1 - 2 * (x * x + y * y)).toFloat(); m[11] = 0f
        m[12] = p.t.x.toFloat(); m[13] = p.t.y.toFloat(); m[14] = p.t.z.toFloat(); m[15] = 1f
    }

    companion object {
        private const val LIT_VS = """#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vN;
out vec3 vW;
void main() {
    vN = mat3(uModel) * aNormal;
    vW = (uModel * vec4(aPos, 1.0)).xyz;
    gl_Position = uMVP * vec4(aPos, 1.0);
}"""

        // Lambert + a little specular, plus a cool rim so the black parts still read
        // against the dark background.
        private const val LIT_FS = """#version 300 es
precision mediump float;
in vec3 vN;
in vec3 vW;
uniform vec4 uColor;
uniform vec3 uLight;
uniform vec3 uEye;
out vec4 outColor;
void main() {
    vec3 v = normalize(uEye - vW);
    // Facet normal as the fallback for a degenerate vertex normal. Then turned to face
    // the eye: the DAEs' winding is not consistent, so gl_FrontFacing cannot be trusted.
    vec3 n = length(vN) > 1e-3 ? normalize(vN) : normalize(cross(dFdx(vW), dFdy(vW)));
    if (dot(n, v) < 0.0) n = -n;
    float d = max(dot(n, normalize(uLight)), 0.0);
    float s = pow(max(dot(n, normalize(normalize(uLight) + v)), 0.0), 40.0);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    vec3 albedo = max(uColor.rgb, vec3(0.11));
    vec3 c = albedo * (0.3 + 0.7 * d) + 0.12 * s + vec3(0.24, 0.84, 0.78) * 0.22 * rim;
    outColor = vec4(c, 1.0);
}"""

        /** Unit cube, 4 vertices per face (position, normal), so each face shades flat. */
        private val CUBE: FloatArray = run {
            val out = ArrayList<Float>()
            for (axis in 0..2) for (sign in listOf(-1f, 1f)) {
                val u = (axis + 1) % 3
                val v = (axis + 2) % 3
                for ((a, b) in listOf(-1f to -1f, 1f to -1f, 1f to 1f, -1f to 1f)) {
                    val p = FloatArray(3)
                    val n = FloatArray(3)
                    p[axis] = 0.5f * sign
                    p[u] = 0.5f * a * sign
                    p[v] = 0.5f * b
                    n[axis] = sign
                    out += p.toList() + n.toList()
                }
            }
            out.toFloatArray()
        }
        private val CUBE_INDICES = ShortArray(36) { i -> (i / 6 * 4 + intArrayOf(0, 1, 2, 0, 2, 3)[i % 6]).toShort() }

        private const val BOX_VS = """#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCentre;
layout(location = 2) in vec3 aNormal;
uniform mat4 uMVP;
uniform float uSize;
out vec3 vLocal;
out vec3 vN;
out float vZ;
void main() {
    vLocal = aPos;
    vN = aNormal;
    vZ = aCentre.z;
    gl_Position = uMVP * vec4(aCentre + aPos * uSize, 1.0);
}"""

        // Height in -0.5..2.0 m through Google's polynomial fit of the turbo colormap, shaded
        // per face, with darker edges so neighbouring boxes stay apart.
        private const val BOX_FS = """#version 300 es
precision highp float;
in vec3 vLocal;
in vec3 vN;
in float vZ;
uniform vec3 uLight;
out vec4 outColor;
vec3 turbo(float x) {
    const vec4 r4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 g4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 b4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 r2 = vec2(-152.94239396, 59.28637943);
    const vec2 g2 = vec2(4.27729857, 2.82956604);
    const vec2 b2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, r4) + dot(v2, r2), dot(v4, g4) + dot(v2, g2), dot(v4, b4) + dot(v2, b2));
}
void main() {
    vec3 a = abs(vLocal) * 2.0;
    float mid = a.x + a.y + a.z - max(a.x, max(a.y, a.z)) - min(a.x, min(a.y, a.z));
    float edge = smoothstep(0.80, 0.94, mid);
    float light = 0.55 + 0.45 * max(dot(vN, normalize(uLight)), 0.0);
    vec3 c = turbo(clamp((vZ + 0.5) / 2.5, 0.0, 1.0)) * light * (1.0 - 0.45 * edge);
    outColor = vec4(c, 1.0);
}"""

        private const val LINE_VS = """#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }"""

        private const val LINE_FS = """#version 300 es
precision mediump float;
uniform vec4 uColor;
out vec4 outColor;
void main() { outColor = uColor; }"""
    }
}

/** RGBA8, 24-bit depth and 4x MSAA, stepping down to what the device has. */
class MsaaConfigChooser : GLSurfaceView.EGLConfigChooser {
    override fun chooseConfig(egl: EGL10, display: EGLDisplay): EGLConfig {
        for ((samples, depth) in listOf(4 to 24, 0 to 24, 4 to 16, 0 to 16)) {
            val attrs = intArrayOf(
                EGL10.EGL_RED_SIZE, 8, EGL10.EGL_GREEN_SIZE, 8, EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, depth, EGL10.EGL_RENDERABLE_TYPE, 0x40, // EGL_OPENGL_ES3_BIT_KHR
                EGL10.EGL_SAMPLE_BUFFERS, if (samples > 0) 1 else 0, EGL10.EGL_SAMPLES, samples,
                EGL10.EGL_NONE,
            )
            val configs = arrayOfNulls<EGLConfig>(1)
            val n = IntArray(1)
            if (egl.eglChooseConfig(display, attrs, configs, 1, n) && n[0] > 0) return configs[0]!!
        }
        error("no EGL config")
    }
}
