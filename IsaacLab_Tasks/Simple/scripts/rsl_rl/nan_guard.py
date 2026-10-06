import torch


def install_nan_guard(env, max_reports: int = 20, max_reward: float = 1e3, max_obs: float = 1e4):
    """Zero non-finite or huge observations and rewards before they reach PPO, and print where they came from.
    A physics blow-up gives finite but huge values too, and one inf loss turns every weight into NaN."""
    inner = env.unwrapped
    step = env.step
    state = {"reports": 0, "steps": 0}

    def guarded_step(actions):
        obs, rew, dones, extras = step(actions)
        state["steps"] += 1
        bad_rew = ~(rew.abs() <= max_reward)
        bad_obs = {k: ~(v.abs() <= max_obs).all(dim=-1) for k, v in obs.items()}
        bad = bad_rew.clone()
        for b in bad_obs.values():
            bad |= b
        if bad.any():
            if state["reports"] < max_reports:
                state["reports"] += 1
                _report(inner, state["steps"], bad, bad_rew, bad_obs, max_reward, max_obs)
            for k, v in obs.items():
                v[bad_obs[k]] = 0.0
            rew[bad] = 0.0
        return obs, rew, dones, extras

    env.step = guarded_step
    return env


def _report(inner, step, bad, bad_rew, bad_obs, max_reward, max_obs):
    ids = bad.nonzero().flatten()
    print(f"[NanGuard] step {step}: {len(ids)} envs non-finite or huge, e.g. {ids[:8].tolist()}", flush=True)
    rm = inner.reward_manager
    terms = [n for i, n in enumerate(rm._term_names) if not (rm._step_reward[ids, i].abs() * inner.step_dt <= max_reward).all()]
    print(f"[NanGuard]   reward bad in {int(bad_rew.sum())} envs, terms {terms}", flush=True)
    om = inner.observation_manager
    for group, mask in bad_obs.items():
        if not mask.any() or group not in om.active_terms:
            continue
        names = []
        for name, cfg in zip(om.active_terms[group], om._group_obs_term_cfgs[group]):
            try:
                value = cfg.func(inner, **cfg.params)
            except Exception:
                continue
            if not (torch.as_tensor(value)[ids].abs() <= max_obs).all():
                names.append(name)
        print(f"[NanGuard]   obs '{group}' bad in {int(mask.sum())} envs, terms {names}", flush=True)
    robot = inner.scene["robot"].data
    for name in ("root_pos_w", "root_lin_vel_w", "joint_pos", "joint_vel"):
        value = getattr(robot, name)
        value = value.torch if hasattr(value, "torch") else value
        sub = value[ids]
        finite = torch.isfinite(sub)
        peak = sub[finite].abs().max().item() if finite.any() else float("nan")
        print(f"[NanGuard]   {name}: finite {finite.all(dim=-1).sum().item()}/{len(ids)}, |max finite| {peak:.3g}", flush=True)
    terrain = getattr(inner.scene, "terrain", None)
    if terrain is not None and getattr(terrain, "terrain_types", None) is not None:
        print(f"[NanGuard]   terrain level {terrain.terrain_levels[ids[:8]].tolist()} type {terrain.terrain_types[ids[:8]].tolist()}", flush=True)
