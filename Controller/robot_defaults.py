"""Constants shared by every process that talks to a quadruped.

The nominal stance lives here because five separate modules used to hardcode their own
copy -- the policy's observation offset, its action offset, the safety fallback pose and
both MuJoCo drivers. They have to stay byte-identical: the policy is trained on
(joint_pos - default) and its actions are applied as (action * scale + default), so a
drift between any two of them silently biases the whole control loop.
"""

import numpy as np

# Isaac Lab's init_state.joint_pos for the Unitree A1, Go1 and Go2 -- all three share the
# same nominal stance, so one constant covers the whole multi-robot training setup.
# Order is Isaac's: all hips, then all thighs, then all calves, each FL, FR, RL, RR.
DEFAULT_STANCE_QPOS = np.array(
    [
        0.1, -0.1, 0.1, -0.1,      # hips
        0.8, 0.8, 1.0, 1.0,        # thighs (front 0.8, rear 1.0)
        -1.5, -1.5, -1.5, -1.5,    # calves
    ],
    dtype=np.float32,
)
