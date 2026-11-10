"""Procedural terrain for the MuJoCo driver.

The menagerie scenes stand the robot on an infinite `plane`, which is the one
surface a policy never has to place a foot carefully on. Real ground is not
flat, and a swing that clears a plane by a couple of millimetres catches on
anything that is not - so a policy validated only on `plane` can look fine in
sim and drag a foot on carpet.

`rough` swaps that plane for a heightfield whose elevation is regenerated every
run, so repeated runs are not the same ground twice. The field is white noise
put through a few smoothing passes: per-cell noise would be spikes narrower
than a foot, which the collision geometry mostly bridges over, whereas smoothed
noise gives slopes and dips a foot actually lands on.

Amplitude is peak-to-peak in metres and is the knob that matters - it wants to
be comparable to the swing clearance the policy achieves, not to the robot's
leg length.

`stairs` puts a wide flight up, a platform and a flight down ahead of the
robot; `narrow` is ClockScan's narrow tile: a flight with open sides up to an
upper floor. Both are boxes on the flat plane, sized by the stair_* keys.
`spikes` is a path of stepping-stone posts over a pit, from a start platform
to an end platform, sized by the spike_* keys: a foot that misses a post top
drops into a gap, so the policy has to place each foot.

Terrain names match the launcher's existing vocabulary ("flat" / "rough") and
arrive via the QUADRUPED_TERRAIN environment variable.
"""

import os
import tempfile

import numpy as np
import mujoco


TERRAINS = ("flat", "rough", "stairs", "narrow", "spikes")
BOX_TERRAINS = ("stairs", "narrow", "spikes")

# The line every unitree_* scene.xml uses for its ground plane. Matched
# verbatim rather than parsed: if upstream changes it, we want a loud failure
# telling us to look, not a silently flat "rough" terrain.
_PLANE_GEOM = ('<geom name="floor" size="0 0 0.05" type="plane" '
               'material="groundplane"/>')

_HFIELD_GEOM = ('<geom name="floor" type="hfield" hfield="noise_terrain" '
                'material="groundplane"/>')

# Defaults, overridable from quadruped_bringup/config/config.yaml under `terrain:`.
#
# Spatial frequency is set by cell size (2*extent/resolution) together with the
# number of blur passes. At 400 cells over 12 m the cells are 3 cm - smaller
# than a foot - and one blur pass leaves features roughly 9 cm across, so the
# robot meets several distinct bumps per stride rather than a slow swell.
#
# smoothing=0 is available but not useful ground: with 3 cm cells it leaves
# steps as large as the full amplitude between neighbours, i.e. vertical walls
# the foot hits edge-on rather than terrain it can stand on.
DEFAULTS = {
    "amplitude": 0.08,   # peak-to-peak elevation (m)
    "extent": 6.0,       # half-width of the field (m); 6.0 -> 12x12 m of ground
    "resolution": 400,   # cells per side; 400 over 12 m -> 3 cm cells
    "smoothing": 1,      # blur passes; higher = broader, gentler features
    "seed": None,        # None = new ground every run
    # Foot contact friction. The foot geoms carry priority=1, so THEIR friction
    # decides every foot-ground contact and the floor geom's value is ignored -
    # which is why this is set on the feet rather than on the ground.
    #
    # Measured on this model: slip under a 45 N lateral push falls from 334 mm
    # at 0.3 to 8 mm at 0.8, then saturates around 1.5-2.0 (what remains is leg
    # and contact compliance, not sliding). Past ~2 nothing changes, so there is
    # no point going higher even though MuJoCo stays stable well beyond it.
    #
    # Reference: rubber on carpet ~0.7-1.2, on concrete ~0.8-1.0.
    "foot_friction_slide": 1.1,
    # Torsional friction resists a foot twisting in place. Carpet resists this
    # far more than a smooth floor; the menagerie default of 0.02 is a smooth
    # floor value.
    "foot_friction_torsion": 0.05,
    # stairs / narrow: step height and depth (m), steps per flight, flight width
    # (m), and the gap from the spawn point to the first riser (m).
    "stair_height": 0.18,
    "stair_depth": 0.30,
    "stair_steps": 6,
    "stair_width": 3.0,
    "narrow_steps": 15,
    "narrow_width": 0.8,
    "stair_gap": 0.6,
    # spikes: post height over the pit, top side, centre-to-centre pitch, length and width of the path (m),
    # random post height (+-) and shift of each post (+-, m; seeded by `seed`).
    "spike_height": 0.40,
    "spike_top": 0.14,
    "spike_pitch": 0.24,
    "spike_length": 3.0,
    "spike_width": 1.0,
    "spike_jitter": 0.02,
    "spike_shift": 0.02,
}

FOOT_GEOMS = ("FL", "FR", "RL", "RR")


def resolve(cfg=None):
    """Merge the `terrain` block of config.yaml over DEFAULTS."""
    out = dict(DEFAULTS)
    if cfg:
        for k, v in (cfg.get("terrain") or {}).items():
            if k in out:
                out[k] = v
    return out


def scene_path(base_scene, terrain, params, tmp_files):
    """Return the scene XML to compile for `terrain`.

    For "flat" this is `base_scene` untouched. For "rough" a copy with the
    plane swapped for a heightfield is written *next to* the original - the
    scene `<include>`s the robot XML and inherits its meshdir, both of which
    resolve relative to the file's own directory, so a copy anywhere else
    would fail to find the meshes.

    Paths written are appended to `tmp_files` for the caller to clean up.
    """
    if terrain not in TERRAINS:
        raise ValueError("unknown terrain %r (expected one of %s)"
                         % (terrain, ", ".join(TERRAINS)))
    if terrain == "flat":
        return base_scene

    xml = open(base_scene, encoding="utf-8").read()
    if terrain in BOX_TERRAINS:
        geoms = "".join(
            '    <geom type="box" pos="%g %g %g" size="%g %g %g" rgba="0.55 0.55 0.6 1"/>\n'
            % (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1), 0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0))
            for x0, x1, y0, y1, z0, z1 in stair_boxes(terrain, params))
        return _write_beside(base_scene, xml.replace("</worldbody>", geoms + "  </worldbody>", 1), tmp_files)
    if _PLANE_GEOM not in xml:
        raise RuntimeError(
            "%s has no recognisable ground plane to replace - upstream may have "
            "changed the floor geom; update _PLANE_GEOM in quadruped_drivers/terrain.py"
            % base_scene)

    hfield = ('  <hfield name="noise_terrain" nrow="%d" ncol="%d" '
              'size="%g %g %g 0.1"/>\n  </asset>'
              % (params["resolution"], params["resolution"],
                 params["extent"], params["extent"], params["amplitude"]))
    xml = xml.replace("</asset>", hfield, 1)
    xml = xml.replace(_PLANE_GEOM, _HFIELD_GEOM, 1)
    return _write_beside(base_scene, xml, tmp_files)


def _write_beside(base_scene, xml, tmp_files):
    fd, path = tempfile.mkstemp(prefix="_terrain_", suffix=".xml",
                                dir=os.path.dirname(base_scene))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(xml)
    tmp_files.append(path)
    return path


def stair_boxes(terrain, params):
    """(x0, x1, y0, y1, z0, z1) boxes of a stairs terrain, the robot spawned at the origin facing +x."""
    if terrain == "spikes":
        return spike_boxes(params)
    h, d, x0 = float(params["stair_height"]), float(params["stair_depth"]), float(params["stair_gap"])
    if terrain == "narrow":
        n, w = int(params["narrow_steps"]), float(params["narrow_width"])
        x1 = x0 + n * d
        boxes = [(x0 + k * d, x0 + (k + 1) * d, -0.5 * w, 0.5 * w, 0.0, (k + 1) * h) for k in range(n)]
        return boxes + [(x1, x1 + 4.0, -2.5, 2.5, 0.0, n * h)]
    n, w = int(params["stair_steps"]), float(params["stair_width"])
    x1 = x0 + n * d
    boxes = [(x0 + k * d, x0 + (k + 1) * d, -0.5 * w, 0.5 * w, 0.0, (k + 1) * h) for k in range(n)]
    boxes.append((x1, x1 + 2.0, -0.5 * w, 0.5 * w, 0.0, n * h))
    return boxes + [(x1 + 2.0 + k * d, x1 + 2.0 + (k + 1) * d, -0.5 * w, 0.5 * w, 0.0, (n - k) * h) for k in range(n)]


def spike_boxes(params):
    """Start platform, the posts, end platform. The robot spawns on the start platform at the origin facing +x."""
    h, top, pitch = float(params["spike_height"]), float(params["spike_top"]), float(params["spike_pitch"])
    x0, length, w = float(params["stair_gap"]), float(params["spike_length"]), float(params["spike_width"])
    jitter, shift = float(params["spike_jitter"]), float(params["spike_shift"])
    rng = np.random.default_rng(params["seed"])
    nx, ny = max(1, int(round(length / pitch))), max(1, int(round(w / pitch)))
    x1 = x0 + nx * pitch
    boxes = [(-1.0, x0, -0.5 * w - 0.5, 0.5 * w + 0.5, 0.0, h)]
    for i in range(nx):
        for j in range(ny):
            cx = x0 + (i + 0.5) * pitch + rng.uniform(-shift, shift)
            cy = (j - 0.5 * (ny - 1)) * pitch + rng.uniform(-shift, shift)
            boxes.append((cx - 0.5 * top, cx + 0.5 * top, cy - 0.5 * top, cy + 0.5 * top,
                          0.0, h + rng.uniform(-jitter, jitter)))
    return boxes + [(x1, x1 + 2.0, -0.5 * w - 0.5, 0.5 * w + 0.5, 0.0, h)]


def spike_end(params):
    """x where the posts end and the end platform starts."""
    pitch = float(params["spike_pitch"])
    return float(params["stair_gap"]) + max(1, int(round(float(params["spike_length"]) / pitch))) * pitch


def spawn_ground(terrain, params):
    """Ground height at the origin, where the driver spawns the robot."""
    return float(params["spike_height"]) if terrain == "spikes" else 0.0


def stair_spawns(terrain, params):
    """{"up", "down"}: (x, y, ground z, yaw) to start a stairs test from, facing the flight."""
    if terrain == "spikes":
        h = float(params["spike_height"])
        return {"up": (0.0, 0.0, h, 0.0), "down": (spike_end(params) + float(params["stair_gap"]), 0.0, h, np.pi)}
    h, d, x0 = float(params["stair_height"]), float(params["stair_depth"]), float(params["stair_gap"])
    if terrain == "narrow":
        n = int(params["narrow_steps"])
        return {"up": (0.0, 0.0, 0.0, 0.0), "down": (x0 + n * d + x0, 0.0, n * h, np.pi)}
    n = int(params["stair_steps"])
    return {"up": (0.0, 0.0, 0.0, 0.0), "down": (x0 + n * d + 1.0, 0.0, n * h, 0.0)}


def apply_foot_friction(model, params):
    """Override the foot geoms' sliding and torsional friction in place.

    Applied after the model is compiled rather than edited into the XML, so the
    vendored menagerie files stay untouched and the value is tunable from
    config.yaml without a scene edit.

    Returns (slide, torsion, n_feet_set); n_feet_set is 0 on a model whose foot
    geoms are not named as expected, which is worth surfacing rather than
    silently leaving stock friction in place.
    """
    slide = float(params["foot_friction_slide"])
    torsion = float(params["foot_friction_torsion"])
    n = 0
    for name in FOOT_GEOMS:
        try:
            gid = model.geom(name).id
        except KeyError:
            continue
        model.geom_friction[gid][0] = slide
        model.geom_friction[gid][1] = torsion
        n += 1
    return slide, torsion, n


def randomize(model, params):
    """Fill the model's heightfield with smoothed noise, normalised to [0, 1].

    Returns a dict describing the ground that was generated, or None on a model
    without a heightfield - so callers do not need to branch on the terrain
    they asked for.

    The reported `step` is the largest height difference between neighbouring
    cells. That is the number to compare against a policy's swing clearance:
    peak-to-peak says how undulating the ground is overall, but it is the step
    between adjacent cells that a toe actually catches on.
    """
    if model.nhfield == 0:
        return None

    rng = np.random.default_rng(params["seed"])
    nrow, ncol = int(model.hfield_nrow[0]), int(model.hfield_ncol[0])
    field = rng.random((nrow, ncol))

    # 3x3 binomial blur. Wrapping via np.roll keeps the field tileable, which
    # also stops the edges of the patch from being systematically lower than
    # the middle after normalisation.
    kern = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
    kern /= kern.sum()
    for _ in range(int(params["smoothing"])):
        field = sum(np.roll(np.roll(field, i - 1, 0), j - 1, 1) * kern[i, j]
                    for i in range(3) for j in range(3))

    field -= field.min()
    peak = field.max()
    if peak > 1e-9:
        field /= peak
    model.hfield_data[:] = field.ravel()

    # hfield_size[2] is the elevation the normalised data is scaled by, so these
    # are true metres of the ground the robot will walk on.
    elev = float(model.hfield_size[0][2])
    steps = np.abs(np.diff(field, axis=0)) * elev
    return {
        "ptp": float(np.ptp(field) * elev),
        "cell": float(2.0 * model.hfield_size[0][0] / ncol),
        "step": float(steps.max()),
        "step_p95": float(np.percentile(steps, 95)),
    }
