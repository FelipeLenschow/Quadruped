"""Commanded-vs-measured velocity arrows for the MuJoCo viewer.

Two arrows float above the trunk from a common origin: the velocity the policy
was asked for, and the one the base is actually carrying. Drawn together they
make the tracking error readable as a shape - a length mismatch is a speed
error, an angle between them is a heading/drift error - instead of something
you have to reconstruct from the printed vx.

Both inputs are BASE-frame (the frame /cmd_vel is interpreted in, and the frame
the driver's `vel_b` is computed in). They are mapped into the world with the
trunk's YAW only, so pitching and rolling while trotting does not tip the arrows
out of the ground plane and cost the comparison its readability. The vertical
velocity component is dropped for the same reason: it is bounce, not tracking.
"""

import numpy as np
import mujoco

TRUNK_BODY_NAMES = ("trunk", "base")

# mjv_connector takes from/to points and owns the arrow's origin-and-length
# convention, so it is preferred over placing the geom by hand. It is named
# mjv_makeConnector on older MuJoCo, and both are absent on older still.
_CONNECTOR = getattr(mujoco, "mjv_connector", getattr(mujoco, "mjv_makeConnector", None))


class VelocityArrowOverlay:
    # Geometry in metres, and metres of arrow per m/s.
    ARROW_LIFT = 0.32      # above the trunk origin, clear of the back
    ARROW_SCALE = 0.35
    ARROW_RADIUS = 0.014
    PAIR_GAP = 0.05        # keeps the two arrows off each other when they agree
    STUB_R = 0.022         # marker drawn in place of an arrow at ~zero speed
    MIN_SPEED = 0.02       # below this the direction is noise, not a heading
    STEM_WIDTH = 0.004

    CMD_RGBA = (0.25, 0.55, 1.00, 0.85)   # blue  - what was asked for
    ACT_RGBA = (1.00, 0.55, 0.10, 0.90)   # amber - what the base is doing

    def __init__(self, model, body_names=TRUNK_BODY_NAMES):
        # Resolved once - draw() runs on every render.
        self.body_id = -1
        for name in body_names:
            b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b_id != -1:
                self.body_id = b_id
                break

    @property
    def available(self):
        """False when the model has no trunk body to hang the arrows over."""
        return self.body_id != -1

    def draw(self, scn, data, cmd_vel, base_lin_vel, reset=True):
        """Draw the commanded and measured velocity arrows above the trunk.

        `scn` is the viewer's user_scn. `cmd_vel` and `base_lin_vel` are
        base-frame velocities; only their x/y components are used, so the
        4-element command vector and the 3-element velocity can be passed
        straight in. `reset` clears the scene first - pass False to compose
        these arrows after another overlay that has already cleared it.
        """
        if scn is None or self.body_id == -1:
            return
        if reset:
            scn.ngeom = 0

        trunk_pos = data.xpos[self.body_id].copy()
        yaw = np.arctan2(data.xmat[self.body_id][3], data.xmat[self.body_id][0])
        c, s = np.cos(yaw), np.sin(yaw)
        anchor = trunk_pos + [0.0, 0.0, self.ARROW_LIFT]

        # Stem tying the arrows to the robot: with a tracking camera the pair
        # would otherwise read as free-floating scenery.
        self._add(scn, mujoco.mjtGeom.mjGEOM_BOX,
                  [self.STEM_WIDTH, self.STEM_WIDTH, self.ARROW_LIFT * 0.5],
                  trunk_pos + [0.0, 0.0, self.ARROW_LIFT * 0.5],
                  (0.85, 0.85, 0.90, 0.25))

        for vel, rgba, dz, tag in (
            (cmd_vel, self.CMD_RGBA, +0.5 * self.PAIR_GAP, "cmd"),
            (base_lin_vel, self.ACT_RGBA, -0.5 * self.PAIR_GAP, "vel"),
        ):
            vx, vy = float(vel[0]), float(vel[1])
            world = np.array([c * vx - s * vy, s * vx + c * vy, 0.0])
            speed = float(np.linalg.norm(world))
            origin = anchor + [0.0, 0.0, dz]
            label = f"{tag} {speed:.2f} m/s"

            # A standing robot and a missing overlay look the same if a zero
            # command simply draws nothing, so zero gets a marker of its own.
            if speed < self.MIN_SPEED:
                self._add(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
                          [self.STUB_R, 0, 0], origin, rgba, label=label)
            else:
                self._add_arrow(scn, origin, origin + world * self.ARROW_SCALE,
                                rgba, label)

    def _add(self, scn, gtype, size, pos, rgba, mat=None, label=None):
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, gtype,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            (np.eye(3) if mat is None else mat).flatten(),
            np.asarray(rgba, dtype=np.float32),
        )
        self._set_label(g, label)
        scn.ngeom += 1
        return g

    def _add_arrow(self, scn, origin, tip, rgba, label=None):
        """Arrow from `origin` to `tip`, however this MuJoCo places arrows."""
        if scn.ngeom >= scn.maxgeom:
            return
        arrow = mujoco.mjtGeom.mjGEOM_ARROW
        origin = np.asarray(origin, dtype=np.float64)
        tip = np.asarray(tip, dtype=np.float64)

        if _CONNECTOR is None:
            # No connector: place it by hand. MuJoCo extends an arrow from
            # `pos` along the geom's local +z by size[2].
            span = tip - origin
            length = float(np.linalg.norm(span))
            self._add(scn, arrow,
                      [self.ARROW_RADIUS, self.ARROW_RADIUS, length],
                      origin, rgba, mat=self._frame_from_dir(span / length),
                      label=label)
            return

        # mjv_connector sets pos/mat/size but not rgba, so the geom is
        # initialised first and then pointed at the tip.
        span = tip - origin
        length = float(np.linalg.norm(span))
        g = self._add(scn, arrow, [self.ARROW_RADIUS, self.ARROW_RADIUS, length], origin, rgba, label=label)
        if g is not None:
            _CONNECTOR(g, arrow, self.ARROW_RADIUS, origin, tip)

    @staticmethod
    def _set_label(g, label):
        if label is None:
            return
        # Labels render only on some viewer builds; the arrows carry the
        # reading on their own, so never fail the frame over one.
        try:
            g.label = label
        except (AttributeError, TypeError, ValueError):
            pass

    @staticmethod
    def _frame_from_dir(direction):
        """Rotation whose local +z - the axis MuJoCo extends an arrow along -
        is `direction`. The other two columns only have to be orthonormal."""
        z = np.asarray(direction, dtype=np.float64)
        helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        return np.column_stack((x, np.cross(z, x), z))
