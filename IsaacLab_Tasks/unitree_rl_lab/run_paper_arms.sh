#!/usr/bin/env bash
# Paper arms for the low-speed dead zone. Run ONE AT A TIME -- two Isaac Sim processes at once
# hits a carb.tasking mutex assertion and kills both.
#
# Each arm differs from the baseline by exactly one thing; everything else (rewards, weights,
# level curriculum, events, terminations, PPO config, sample budget) is identical.
set -euo pipefail
cd "$(dirname "$0")"

ITERS=${ITERS:-3000}          # c* is saturated by 3000; U0's own default is 50000
SEED=${SEED:-1}

run () {
  echo "=== $1  seed $SEED  $ITERS iters ==="
  python scripts/rsl_rl/train.py --task="$1" --headless --seed="$SEED" --max_iterations="$ITERS"
}

run Unitree-Go2-Velocity-Coverage   # arm C  -- command coverage
run Unitree-Go2-Velocity-Sigma      # arm S  -- reward shape
run Unitree-Go2-Velocity-Both       # arm SC -- both
