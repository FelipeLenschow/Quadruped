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

Terrain names match the launcher's existing vocabulary ("flat" / "rough") and
arrive via the QUADRUPED_TERRAIN environment variable.
"""

import os
import tempfile

import numpy as np
import mujoco


TERRAINS = ("flat", "rough")

# The line every unitree_* scene.xml uses for its ground plane. Matched
# verbatim rather than parsed: if upstream changes it, we want a loud failure
# telling us to look, not a silently flat "rough" terrain.
_PLANE_GEOM = ('<geom name="floor" size="0 0 0.05" type="plane" '
               'material="groundplane"/>')

_HFIELD_GEOM = ('<geom name="floor" type="hfield" hfield="noise_terrain" '
                'material="groundplane"/>')

# Defaults, overridable from Configs/config.yaml under `terrain:`.
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
    if _PLANE_GEOM not in xml:
        raise RuntimeError(
            "%s has no recognisable ground plane to replace - upstream may have "
            "changed the floor geom; update _PLANE_GEOM in Mujoco/terrain.py"
            % base_scene)

    hfield = ('  <hfield name="noise_terrain" nrow="%d" ncol="%d" '
              'size="%g %g %g 0.1"/>\n  </asset>'
              % (params["resolution"], params["resolution"],
                 params["extent"], params["extent"], params["amplitude"]))
    xml = xml.replace("</asset>", hfield, 1)
    xml = xml.replace(_PLANE_GEOM, _HFIELD_GEOM, 1)

    fd, path = tempfile.mkstemp(prefix="_terrain_", suffix=".xml",
                                dir=os.path.dirname(base_scene))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(xml)
    tmp_files.append(path)
    return path


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
