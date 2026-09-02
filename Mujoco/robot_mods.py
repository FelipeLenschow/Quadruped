import numpy as np
import mujoco


DEFAULTS = {
    "foot_mass": 0.0,   # extra mass per foot (kg); 0.0 = stock model
}

# Calf body per leg, and the geom whose position marks the contact point.
_CALF_BODIES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
_FOOT_GEOMS = ("FL", "FR", "RL", "RR")


def resolve(cfg=None):
    """Merge the `robot_mods` block of config.yaml over DEFAULTS."""
    out = dict(DEFAULTS)
    if cfg:
        for k, v in (cfg.get("robot_mods") or {}).items():
            if k in out:
                out[k] = v
    return out


def _parallel_axis(mass, d):
    """Inertia contribution of `mass` offset by `d` from the reference point."""
    return mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def add_foot_mass(model, params):
    """Add a point mass at each foot, updating the calf's mass, CoM and inertia.

    MuJoCo stores inertia as principal moments (`body_inertia`) in a frame given
    by `body_iquat`, about a centre at `body_ipos`. The tensor is rebuilt in the
    body frame, re-referenced to the new centre of mass, and written back in the
    compiled inertial frame - `body_iquat` is left untouched because MuJoCo's
    collision bounding boxes are stored in it.

    Returns a dict of before/after figures, or None when `foot_mass` is 0.
    """
    extra = float(params["foot_mass"])
    if extra <= 0.0:
        return None

    swing_before = swing_after = 0.0
    n = 0
    for body_name, geom_name in zip(_CALF_BODIES, _FOOT_GEOMS):
        try:
            bid = model.body(body_name).id
            gid = model.geom(geom_name).id
        except KeyError:
            continue

        m1 = float(model.body_mass[bid])
        c1 = np.array(model.body_ipos[bid], dtype=float)

        # Foot geom position is stored relative to its parent body, which is
        # the calf - the same frame body_ipos lives in.
        p2 = np.array(model.geom_pos[gid], dtype=float)

        # Principal moments -> full tensor in the body frame, about c1.
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.body_iquat[bid])
        R = R.reshape(3, 3)
        I1 = R @ np.diag(np.array(model.body_inertia[bid], dtype=float)) @ R.T

        m_new = m1 + extra
        c_new = (m1 * c1 + extra * p2) / m_new

        # Shift both contributions onto the new centre of mass. The point mass
        # has no inertia of its own, so it contributes only its offset term.
        I_new = (I1 + _parallel_axis(m1, c1 - c_new)
                 + _parallel_axis(extra, p2 - c_new))

        # Symmetrise: the arithmetic above is symmetric in exact maths.
        I_new = 0.5 * (I_new + I_new.T)

        # Project back onto the compiled inertial frame rather than
        # re-diagonalising into a new one - see the module docstring for why
        # body_iquat must not move.
        inertia_new = np.diag(R.T @ I_new @ R)

        # Swing inertia about the knee (body origin), the number that says how
        # much harder this leg is to throw forward.
        swing_before += float(np.trace(I1 + _parallel_axis(m1, c1)) / 2.0)
        swing_after += float(np.trace(I_new + _parallel_axis(m_new, c_new)) / 2.0)

        model.body_mass[bid] = m_new
        model.body_ipos[bid] = c_new
        model.body_inertia[bid] = inertia_new
        n += 1

    if n == 0:
        return None
    return {
        "per_foot": extra,
        "legs": n,
        "total_added": extra * n,
        "swing_inertia_before": swing_before / n,
        "swing_inertia_after": swing_after / n,
    }
