"""Wheel coordination, spinal-to-cerebellum frame decoupling, and timing for grounded morphing."""
import mujoco as mj
import numpy as np


def buffered_phi(t, duration=6.0, ramp=0.45):
    """Three unit intervals, each with two half-cosine speed ramps."""
    segment = duration/3
    if duration <= 0 or not 0 <= ramp <= segment/2:
        raise ValueError("Each phase must allow time for both ramps")
    t = float(np.clip(t, 0, duration))
    k = min(int(t/segment), 2)
    u = np.clip(t-k*segment, 0, segment)
    if ramp == 0:
        return k+u/segment
    v = 1/(segment-ramp)
    def start(s):
        return .5*v*(s-ramp/np.pi*np.sin(np.pi*s/ramp))
    if u < ramp:
        return k+start(u)
    if u > segment-ramp:
        return k+1-start(segment-u)
    return k+v*(u-ramp/2)


def spinal_morph_kinematics(e, dq):
    """Spinal-to-cerebellum internal deformation kinematics in Virtual Robot Frame {B}.

    Removes common rigid planar translation and wheel-array rotation omega_0 so the
    returned wheel velocities and swerve-stator yaw rates represent pure internal
    reconfiguration, invariant to the choice of kinematic root block.

    Returns:
        relative: (4, 2) world-aligned internal morphing velocities of the 4 axles (m/s).
        stator_morph_yaw_rate: (4,) internal yaw rate of each swerve mounting stator
            projected onto its steering joint axis (rad/s), in {B}.
        omega_morph_common: scalar common planar rotation rate of the wheel array
            relative to the kinematic root block (rad/s).
    """
    m, d = e.model, e.data
    block_dofs = m.jnt_dofadr[e.block_ids]
    centers = d.xanchor[e.wheel_ids, :2]
    r = centers - centers.mean(axis=0)
    tangent = np.column_stack((-r[:, 1], r[:, 0]))
    inertia = max(float(np.sum(r * r)), 1e-9)

    local_motion = np.empty((len(e.wheel_ids), 2))
    jacp = np.zeros((3, m.nv))
    for i, j in enumerate(e.wheel_ids):
        mj.mj_jac(m, d, jacp, None, d.xanchor[j], m.jnt_bodyid[j])
        local_motion[i] = jacp[:2, block_dofs] @ dq

    relative = local_motion - local_motion.mean(axis=0)
    omega_morph_common = float((tangent * relative).sum() / inertia)
    relative -= tangent * omega_morph_common

    stator_morph_yaw_rate = np.empty(len(e.steering_ids))
    jacr = np.zeros((3, m.nv))
    for i, sj in enumerate(e.steering_ids):
        stator = m.body_parentid[m.jnt_bodyid[sj]]
        mj.mj_jacBody(m, d, None, jacr, stator)
        w_rel = jacr[:, block_dofs] @ dq
        w_internal = np.array([0.0, 0.0, -omega_morph_common]) + w_rel
        stator_morph_yaw_rate[i] = float(np.dot(d.xaxis[sj], w_internal))

    return relative, stator_morph_yaw_rate, omega_morph_common


def coordinate_morph_wheels(e, dq):
    """Set steering and hub targets for one physics tick of a grounded morph.

    dq is the eight-axis reference rate in radians/second, ordered dof_1..8.
    Remove the common planar twist so this mode requests no chassis travel.
    """
    relative, stator_yaw_rate, _ = spinal_morph_kinematics(e, dq)
    _steer_and_roll(e, relative, slew=2.2, stator_yaw_rate=stator_yaw_rate)


def _steer_and_roll(
    e,
    v,
    *,
    moving=False,
    slew=2.2,
    heading_velocity=None,
    stator_yaw_rate=None,
    tau=0.05,
    tau_coherence=0.16,
    v0_move=0.055,
    v0_still=0.006,
    slew_cusp=2.6,
    ff_v0_move=0.075,
    unwind_bias=0.45,
):
    """Urgency-gated projective-circle (RP^1) swerve tracking with block-yaw decoupling.

    1. Soft-saturated double-angle (nematic) filter invariant under v -> -v:
       n(g) = [gx^2 - gy^2, 2*gx*gy] / (||g||^2 + v0^2)
       Eliminates quadratic corner freeze-then-whip while remaining C^infty at g = 0.
    2. Directional coherence (nematic vector length of unit guide) + wheel speed gate
       urgency in [0, 1]:
       - High urgency (steady direction OR high speed): crisp zero-lag shortest-line
         position tracking + full mounting-block yaw feedforward (-w_mount * dt).
       - Low urgency (low/moderate-speed transient cusp or inter-stage pause near phi=2):
         smooth lateral-scrub energy gradient sin(2*e) + speed/alignment-scaled block FF
         and gentle joint unwinding so wheels ride WITH rotating upper blocks instead of
         winding up to +-180 deg or spinning 180 deg across brief perpendicular pauses.
    3. Signed cosine-projected hub rolling: seamless forward/reverse rolling without
       sideways tire drag.
    """
    m, d = e.model, e.data
    axes = d.xaxis[e.wheel_ids]
    rolling = np.column_stack((axes[:, 1], -axes[:, 0]))
    norm = np.linalg.norm(rolling, axis=1)
    rolling /= np.maximum(norm, 1e-8)[:, None]
    heading = np.arctan2(rolling[:, 1], rolling[:, 0])

    steer = e.steering_actuators
    qa = e.joint_positions(e.steering_ids)
    guide = v if heading_velocity is None else heading_velocity
    g_mag = np.linalg.norm(guide, axis=1)
    g2 = g_mag * g_mag

    v0 = v0_move if moving else v0_still
    denom = g2 + v0 * v0
    nem_inst = np.column_stack((
        (guide[:, 0] * guide[:, 0] - guide[:, 1] * guide[:, 1]) / denom,
        (2.0 * guide[:, 0] * guide[:, 1]) / denom,
    ))
    safe_g2 = np.maximum(g2, 1e-12)
    unit_inst = np.where(
        (g_mag > 0.002)[:, None],
        np.column_stack((
            (guide[:, 0] * guide[:, 0] - guide[:, 1] * guide[:, 1]) / safe_g2,
            (2.0 * guide[:, 0] * guide[:, 1]) / safe_g2,
        )),
        0.0,
    )

    state = e._wheel_heading
    if state is None or abs(state["next_time"] - d.time) > e.dt * 1e-3:
        state = dict(
            nematic=nem_inst.copy(),
            unit_f=unit_inst.copy(),
            speed_f=g_mag.copy(),
            active=np.zeros(len(v), dtype=bool),
        )
        e._wheel_heading = state

    state["next_time"] = d.time + e.dt
    state["nematic"] += (-np.expm1(-e.dt / tau)) * (nem_inst - state["nematic"])
    a_coh = -np.expm1(-e.dt / tau_coherence)
    has_vel = g_mag > 0.002
    state["unit_f"] = np.where(
        has_vel[:, None],
        state["unit_f"] + a_coh * (unit_inst - state["unit_f"]),
        state["unit_f"] * (1.0 - a_coh),
    )
    state["speed_f"] += (-np.expm1(-e.dt / 0.06)) * (g_mag - state["speed_f"])
    state["active"] = state["speed_f"] > np.where(state["active"], 0.005, 0.012)

    nx = state["nematic"][:, 0]
    ny = state["nematic"][:, 1]
    n_mag = np.hypot(nx, ny)
    coh = np.clip(np.hypot(state["unit_f"][:, 0], state["unit_f"][:, 1]), 0.0, 1.0)
    steady = np.clip((coh - 0.82) / 0.14, 0.0, 1.0)

    # Projective shortest-line error err_line in [-pi/2, +pi/2] directly from physical heading
    c2 = np.cos(2.0 * heading)
    s2 = np.sin(2.0 * heading)
    sin2_w = ny * c2 - nx * s2
    cos2_w = nx * c2 + ny * s2
    err_line = 0.5 * np.arctan2(sin2_w, cos2_w)

    # Deterministic, symmetric saddle tie-breaker near exact +-90 deg (no jitter sensitivity)
    w_mount = np.zeros(len(steer)) if stator_yaw_rate is None else np.asarray(stator_yaw_rate, dtype=float)
    near_90 = np.abs(err_line) > np.radians(85.0)
    tie_sign = np.where(
        np.abs(qa) > np.radians(8.0),
        -np.sign(qa),
        np.where(
            np.abs(w_mount) > 0.05,
            np.sign(w_mount),
            np.where(np.cos(heading) < 0.0, 1.0, -1.0),
        ),
    )
    tie_sign = np.where(tie_sign == 0.0, 1.0, tie_sign)
    err_sym = np.where(near_90, tie_sign * np.abs(err_line), err_line)

    upright = d.xaxis[e.steering_ids, 2] > 0.65

    if not moving:
        ctrl_ff = d.ctrl[steer] + np.where(upright, -w_mount * e.dt, 0.0)
        desired = qa + np.clip(np.where(state["active"] & (n_mag > 0.04), err_sym, 0.0), -0.06, 0.06)
        limit = slew * e.dt
        d.ctrl[steer] = ctrl_ff + np.where(upright, np.clip(desired - ctrl_ff, -limit, limit), 0.0)
    else:
        v_metric = np.maximum(state["speed_f"], g_mag)
        high_spd = np.clip((v_metric - 0.095) / 0.050, 0.0, 1.0)
        urgency = np.maximum(steady, high_spd)

        sin2_unit = np.sin(2.0 * err_line)
        mis = np.clip(0.5 * (1.0 - np.cos(2.0 * err_line)), 0.0, 1.0)
        delta_cusp = 0.5 * sin2_unit - (unwind_bias * 0.15) * mis * np.tanh(qa / np.radians(60.0))
        ok = state["active"] & (n_mag > 0.10)
        delta_eff = np.where(ok, (1.0 - urgency) * delta_cusp + urgency * err_sym, 0.0)

        slew_grad = slew_cusp * n_mag * np.abs(np.tanh(2.2 * sin2_unit)) + (unwind_bias * mis)
        slew_eff = (1.0 - urgency) * slew_grad + urgency * slew

        v_frac = (state["speed_f"] ** 2) / (state["speed_f"] ** 2 + ff_v0_move * ff_v0_move)
        align_frac = np.maximum(np.cos(err_line), 0.0)
        gamma = np.maximum(urgency, v_frac * align_frac)

        ctrl_ff = d.ctrl[steer] + np.where(upright, -gamma * w_mount * e.dt, 0.0)
        desired = qa + np.clip(delta_eff, -0.06, 0.06)
        limit = slew_eff * e.dt
        d.ctrl[steer] = ctrl_ff + np.where(upright, np.clip(desired - ctrl_ff, -limit, limit), 0.0)

    # Signed cosine-projected hub rolling: seamless forward/reverse rolling, zero sideways drag at 90 deg.
    v_proj = (v * rolling).sum(axis=1)
    speed = np.where(state["active"] & upright, v_proj, 0.0)
    speed /= 0.028 * np.maximum(norm, 0.3)

    lower, upper = m.actuator_forcerange[e.wheel_actuators].T
    torque = 0.05 * (speed - e.joint_velocities(e.wheel_ids))
    if moving:
        damping = m.dof_damping[m.jnt_dofadr[e.wheel_ids]]
        torque += 0.45 * damping * speed
    tau_cmd = np.clip(torque, lower, upper)
    d.ctrl[e.wheel_actuators] = (
        e.joint_positions(e.wheel_ids) + tau_cmd / m.actuator_gainprm[e.wheel_actuators, 0]
    )


def upper_body_center(env):
    """Mass center of the robot above its four passive support hinges, in world m."""
    m, d = env.model, env.data
    roots = np.r_[env.base_body, m.jnt_bodyid[env.pivot_ids]]
    masses = m.body_subtreemass[roots].copy()
    masses[1:] *= -1
    return masses @ d.subtree_com[roots] / masses.sum()


def command_moving_morph(env, angles, rates, twist):
    """One physics tick: world XY velocity of upper COM and planar array yaw rate."""
    m, d = env.model, env.data
    act = env.block_actuators
    actual = env.joint_positions(env.block_ids)
    actual_rate = env.joint_velocities(env.block_ids)
    # Use bounded servo compensation inside both the CAD angle and torque limits.
    correction = 0.75 * (angles - actual) + 0.015 * (rates - actual_rate)
    d.ctrl[act] = np.clip(angles + correction, *m.actuator_ctrlrange[act].T)

    centers = d.xanchor[env.wheel_ids, :2]
    center = upper_body_center(env)[:2]
    lever = centers - center
    around_upper = np.column_stack((-lever[:, 1], lever[:, 0]))
    relative, stator_morph_yaw_rate, _ = spinal_morph_kinematics(env, rates)

    state = env._morph_motion
    if state is None or abs(state["time"] - d.time) > env.dt * 1e-3:
        state = dict(target_xy=center.copy(), yaw_error=0.0, previous=centers.copy(), driving=False)
        env._morph_motion = state
    state["driving"] = state["driving"] or bool(np.any(twist))
    previous = state["previous"]
    mid = (previous + centers) / 2
    mid -= mid.mean(0)
    dyaw = np.sum(np.column_stack((-mid[:, 1], mid[:, 0])) * (centers - previous)) / max(np.sum(mid * mid), 1e-9)
    state["yaw_error"] -= dyaw
    last_center = state.get("last_center", center)
    measured_velocity = (center - last_center) / env.dt
    state["last_center"] = center.copy()
    velocity = (
        twist[:2]
        + 0.1 * (twist[:2] - measured_velocity)
        + np.clip(5 * (state["target_xy"] - center), -0.09, 0.09)
    )
    yaw_rate = twist[2] + np.clip(3 * state["yaw_error"], -0.55, 0.55)
    state["target_xy"] += env.dt * twist[:2]
    state["yaw_error"] += env.dt * twist[2]
    state["previous"] = centers.copy()
    state["time"] = d.time + env.dt

    # Total world stator yaw rate = whole-body yaw rate + spinal internal stator yaw rate
    stator_world_yaw_rate = stator_morph_yaw_rate + (
        yaw_rate * d.xaxis[env.steering_ids, 2] if state["driving"] else 0.0
    )

    if state["driving"]:
        # The planned world velocity sets the heading. Carry most of the fast
        # tracking correction in rolling speed to avoid chasing lateral noise.
        planned = relative + twist[:2] + twist[2] * around_upper
        corrected = relative + velocity + yaw_rate * around_upper
        guide = planned + 0.25 * (corrected - planned)
        _steer_and_roll(
            env,
            corrected,
            moving=True,
            slew=4.0,
            heading_velocity=guide,
            stator_yaw_rate=stator_world_yaw_rate,
        )
    else:
        # Without a travel command let the upper assembly settle during the fold.
        # Anchor the travel reference when the first driving command arrives.
        state["target_xy"] = center.copy()
        state["yaw_error"] = 0.0
        _steer_and_roll(
            env,
            relative,
            slew=2.2,
            stator_yaw_rate=stator_world_yaw_rate,
        )


def moving_phi(t, duration=6.0):
    """Spend 1/4, 1/3, and 5/12 of the sweep on its three folding stages."""
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(t):
        raise ValueError("Expected a positive duration and finite time")
    edges = duration * np.array([0.0, 0.25, 7 / 12, 1.0])
    stage = min(2, np.searchsorted(edges[1:], t, side="right"))
    period = edges[stage + 1] - edges[stage]
    return stage + buffered_phi(t - edges[stage], 3 * period, min(0.45, 0.36 * period)) if t < edges[-1] else 3.0
