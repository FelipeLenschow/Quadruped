"""Foot FSR overlay shared by the MuJoCo driver and the passive twin viewer.

The overlay draws, above each foot, a bar that is linear in raw FSR counts above
that foot's own no-load offset, plus a tick at `contact_threshold`. The gate is
only a few counts of margin, so a foot can hover right at the line and drop out
of contact without anything else in the scene showing it - the tick is there so
that margin is visible instead of inferred.

Both consumers feed it the SAME raw counts: the driver from its simulated FSR,
the twin from /sensors/foot_force (published in raw units by both the MuJoCo
driver and the real robot's driver), so what the viewer shows lines up with
what the policy is actually gated on.

The binary flags come from the driver too - both viewers read them off
/estimator/feet_contact - because the gate that decides them is the driver's,
read from the driver's config.yaml, and off-robot that is a different file than
this process sees. The threshold and the per-foot offsets here only draw the
bars; a driver broadcasting /sensors/foot_force_calibration replaces them, and
thresholding raw counts locally is a last-resort fallback.

The bars are drawn as MARGIN - counts above each foot's own no-load offset -
because the four sensors do not share a zero. One absolute cut across all four
is a different gate on every foot, which is why the offsets are per foot and
the threshold is counts above them.
"""

import numpy as np
import mujoco

from Configs.config_loader import load_config

FOOT_NAMES = ("FL", "FR", "RL", "RR")


class FootContactOverlay:
    # Bar geometry, in metres. BAR_SPAN counts fill BAR_HEIGHT, and the
    # threshold tick is drawn at its own height on the same scale.
    BAR_HEIGHT = 0.18
    BAR_SPAN = 28.0       # raw counts from bias to the top of the bar
    BAR_WIDTH = 0.010
    MARKER_R = 0.038
    BAR_LIFT = 0.048      # clears the marker, so a below-threshold bar still shows

    def __init__(self, model, config=None, foot_names=FOOT_NAMES,
                 site_fmt="{}_foot"):
        cfg = load_config() if config is None else config
        est_cfg = cfg.get("state_estimator", {})
        self.contact_threshold = float(est_cfg.get("contact_threshold", 10.0))
        self.foot_names = list(foot_names)
        # Stand-in until a driver broadcasts what it actually measured.
        self.fsr_offset = np.full(len(self.foot_names),
                                  float(est_cfg.get("fsr_bias", 16.0)))

        # Foot site ids resolved once - draw() runs on every render.
        self.foot_site_id = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_fmt.format(f))
            for f in self.foot_names
        ]

    def set_calibration(self, threshold, offsets):
        """Adopt a driver's gate. Returns True when it differs from the current one."""
        thr = float(threshold)
        off = np.asarray(offsets, dtype=np.float64)
        if off.size != self.fsr_offset.size:
            return False
        if (abs(thr - self.contact_threshold) < 1e-6
                and np.allclose(off, self.fsr_offset)):
            return False
        self.contact_threshold = thr
        self.fsr_offset = off
        return True

    @property
    def available(self):
        """False when the model has no foot sites to hang the bars on."""
        return any(s != -1 for s in self.foot_site_id)

    def draw(self, scn, data, raw_counts, contact=None):
        """Draw a contact marker and an FSR bar over each foot.

        `scn` is the viewer's user_scn, `raw_counts` the four raw FSR readings
        in FL, FR, RL, RR order - raw, not offset-corrected, since the offsets
        are subtracted here. `contact` carries the driver's own flags; left out,
        the margin is thresholded here as a fallback.
        """
        if scn is None:
            return
        scn.ngeom = 0

        raw_counts = np.asarray(raw_counts, dtype=np.float64)
        margin = raw_counts - self.fsr_offset
        if contact is None:
            contact = margin > self.contact_threshold
        contact = np.asarray(contact, dtype=np.float64)

        def add(gtype, size, pos, rgba, mat=None):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, gtype,
                np.asarray(size, dtype=np.float64),
                np.asarray(pos, dtype=np.float64),
                (np.eye(3) if mat is None else mat).flatten(),
                np.asarray(rgba, dtype=np.float32),
            )
            scn.ngeom += 1

        thr_frac = np.clip(self.contact_threshold / self.BAR_SPAN, 0.0, 1.0)

        for i, s_id in enumerate(self.foot_site_id):
            if s_id == -1:
                continue
            base = data.site_xpos[s_id].copy()

            in_contact = contact[i] > 0.5
            frac = np.clip(margin[i] / self.BAR_SPAN, 0.0, 1.0)

            # Marker on the foot itself: green in contact, dark red when the
            # policy is being told the foot is airborne.
            color = (0.1, 0.9, 0.2, 0.85) if in_contact else (0.7, 0.1, 0.1, 0.55)
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [self.MARKER_R, 0, 0], base, color)

            # The bar starts above the marker; a foot sitting just under the
            # threshold has a short bar, and it must not be hidden inside it.
            bar_base = base + [0, 0, self.BAR_LIFT]

            # Empty bar (full span), so the threshold tick has context.
            add(mujoco.mjtGeom.mjGEOM_BOX,
                [self.BAR_WIDTH, self.BAR_WIDTH, self.BAR_HEIGHT * 0.5],
                bar_base + [0, 0, self.BAR_HEIGHT * 0.5],
                (0.85, 0.85, 0.9, 0.30))

            # Filled portion, same colour as the marker. Slightly narrower than
            # the outline so both stay readable where they overlap.
            h = self.BAR_HEIGHT * frac
            if h > 1e-4:
                add(mujoco.mjtGeom.mjGEOM_BOX,
                    [self.BAR_WIDTH * 0.7, self.BAR_WIDTH * 0.7, h * 0.5],
                    bar_base + [0, 0, h * 0.5],
                    color)

            # Threshold tick: a wide, bright slab at contact_threshold. A bar
            # hovering right at this line is a foot about to drop out.
            add(mujoco.mjtGeom.mjGEOM_BOX,
                [self.BAR_WIDTH * 2.4, self.BAR_WIDTH * 2.4, 0.004],
                bar_base + [0, 0, self.BAR_HEIGHT * thr_frac],
                (1.0, 0.85, 0.1, 0.95))
