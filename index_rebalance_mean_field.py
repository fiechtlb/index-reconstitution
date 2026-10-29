"""Diagonal mean-field paths with exact transitions and a stable BVP fallback."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import expm

from _index_rebalance_common import MeanFieldParameters


def _phase_bvp(drift, loading, boundaries, flows, initial, psi):
    """Solve separated constant-flow phases without forward shooting.

    Each phase has its own three state coordinates on [0,1]. Continuity,
    initial inventory/impact, and terminal adjoint conditions determine the
    boundary-value solution. Collocation avoids subtracting exponentially
    growing forward solutions on long horizons.
    """
    durations = np.diff(boundaries)
    phases = len(durations)
    forcing = np.asarray(flows)[:, None] * loading

    def rhs(s, values):
        states = values.reshape(phases, 3, -1)
        result = np.einsum("ij,pjn->pin", drift, states) + forcing[:, :, None]
        return (durations[:, None, None] * result).reshape(3 * phases, -1)

    def boundary(left, right):
        return np.r_[left[[0, 2]] - initial, right[-2] - psi * right[-3],
                     *(right[3*j:3*j+3] - left[3*j+3:3*j+6]
                       for j in range(phases - 1))]

    mesh = np.linspace(0, 1, 41)
    solution = solve_bvp(rhs, boundary, mesh, np.zeros((3 * phases, len(mesh))),
                         tol=1e-9, bc_tol=1e-11, max_nodes=20000)
    boundary_error = np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1])))
    scale = max(1.0, float(np.max(np.abs(solution.y))), float(abs(psi)))
    if (not solution.success or not np.all(np.isfinite(solution.y))
            or np.max(solution.rms_residuals) > 2e-9 or boundary_error > 1e-10 * scale):
        raise RuntimeError("Mean-field collocation failed its ODE/boundary residual checks: "
                           f"{solution.message}")

    def evaluate(times):
        times = np.asarray(times)
        indices = np.minimum(np.searchsorted(boundaries, times, side="right") - 1,
                             phases - 1)
        coordinates = (times - boundaries[indices]) / durations[indices]
        values = solution.sol(coordinates).reshape(phases, 3, -1)
        return values[indices, :, np.arange(len(times))]

    return evaluate


def _stable_scalar_paths(params, drift, loading, flow, psi, times, selected):
    boundaries = np.array([0, params.T_ann, params.T_twap, params.T_eff, params.T])
    expected_flow = params.average_prior @ flow
    expected = _phase_bvp(drift, loading, boundaries, [0, 0, expected_flow, 0],
                          [0, 0], psi)
    announcement = expected(np.array([params.T_ann]))[0]
    # Linearity permits every announced scenario to share these two solves.
    inherited = _phase_bvp(drift, loading, boundaries[1:], [0, 0, 0],
                           announcement[[0, 2]], psi)
    unit_flow = _phase_bvp(drift, loading, boundaries[1:], [0, 1, 0], [0, 0], psi)
    matching = (inherited(np.array([params.T_ann]))[0, 1]
                + expected_flow * unit_flow(np.array([params.T_ann]))[0, 1])
    scale = max(1.0, float(abs(matching)), float(abs(announcement[1])))
    if abs(matching - announcement[1]) > 1e-8 * scale:
        raise RuntimeError("Mean-field announcement adjoint matching failed.")
    result = np.empty((len(selected), len(times), 3))
    pre = times <= params.T_ann
    result[:, pre] = expected(times[pre])[None, :, :]
    if np.any(~pre):
        result[:, ~pre] = (inherited(times[~pre])[None, :, :]
                           + flow[selected, None, None] * unit_flow(times[~pre])[None, :, :])
    return result


def _exact_scalar_paths(params, drift, loading, flow, psi, times, selected):
    """Return checked exact paths, or request collocation if shooting loses accuracy."""
    # Stable decay modes need not be small. Only growing modes cause the large
    # cancellation that makes direct shooting unreliable on long horizons.
    if float(np.max(np.linalg.eigvals(drift).real)) * params.T > 8:
        return None
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = drift
    augmented[:3, 3] = loading

    def forcing_response(t):
        duration = max(0.0, min(t, params.T_eff) - params.T_twap)
        return (expm(drift * max(0.0, t - params.T_eff))
                @ expm(augmented * duration)[:3, 3])

    terminal = np.array([-psi, 1, 0])
    terminal_force = terminal @ forcing_response(params.T)
    denominator = terminal @ expm(drift * params.T)[:, 1]
    transition = expm(drift * (params.T - params.T_ann))
    post_denominator = terminal @ transition[:, 1]
    if not np.all(np.isfinite([terminal_force, denominator, post_denominator])):
        return None
    if denominator == 0 or post_denominator == 0:
        return None
    initial_z = -terminal_force * (params.average_prior @ flow) / denominator
    announcement = expm(drift * params.T_ann)[:, 1] * initial_z
    inherited = announcement.copy()
    inherited[1] = 0
    post_z = -(terminal @ transition @ inherited + terminal_force * flow) / post_denominator
    post_initial = np.tile(announcement, (len(selected), 1))
    post_initial[:, 1] = post_z[selected]

    def evaluate(t):
        if t <= params.T_ann:
            return np.tile(expm(drift * t)[:, 1] * initial_z, (len(selected), 1))
        return (post_initial @ expm(drift * (t - params.T_ann)).T
                + flow[selected, None] * forcing_response(t))

    result = np.asarray([evaluate(t) for t in times]).transpose(1, 0, 2)
    terminal_state = evaluate(params.T)
    scale = max(1.0, float(np.max(np.abs(result))), float(np.max(np.abs(terminal_state))))
    matching_error = abs(params.average_prior @ post_z - announcement[1])
    terminal_error = np.max(np.abs(terminal_state @ terminal)) / max(1.0, abs(psi))
    if (not np.all(np.isfinite(result)) or matching_error > 1e-10 * scale
            or terminal_error > 1e-10 * scale):
        return None
    # Verify local ODE propagation independently of the global shooting formula.
    # This detects cancellation that a terminal or finiteness check alone misses.
    events = np.array([0, params.T_ann, params.T_twap, params.T_eff, params.T])
    for left, right in zip(events[:-1], events[1:]):
        start = left + (right - left) / 3
        step = min((right - left) / 3, 1 / max(1.0, np.linalg.norm(drift, ord=np.inf)))
        local = np.zeros((4, 4))
        local[:3, :3] = drift
        active = params.T_twap <= start < params.T_eff
        local[:3, 3] = loading if active else 0
        transition = expm(local * step)
        # Before the announcement, the expected state is common with zero g.
        predicted = evaluate(start) @ transition[:3, :3].T
        if active:
            predicted += flow[selected, None] * transition[:3, 3]
        if np.max(np.abs(predicted - evaluate(start + step))) > 1e-10 * scale:
            return None
    return result


def solve_diagonal_mean_field_paths(
    *,
    params: MeanFieldParameters,
    times: np.ndarray,
    scenario_numbers: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Solve common mean-field paths on any increasing observation grid.

    The transformed coordinates (X,z=P X+ell,I) satisfy z'=-Phi X and
    z(T)=Psi X(T). Exact phase transitions are retained when direct shooting
    passes conditioning and residual checks. Otherwise separated-phase
    collocation solves the same boundary-value equations stably.
    """
    for name in ("Lambda", "R", "Gamma", "Phi", "Psi"):
        matrix = getattr(params, name)
        if np.any(matrix != np.diag(np.diag(matrix))):
            raise ValueError(f"{name} must be diagonal.")
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > params.T):
        raise ValueError("times must be increasing, finite, and contained in [0,T].")
    if not scenario_numbers or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
        or not 1 <= k <= params.K for k in scenario_numbers
    ):
        raise ValueError("scenario_numbers must contain indices from 1 through K.")
    selected = np.asarray(scenario_numbers) - 1
    inventory = np.empty((len(selected), len(times), params.M))
    impact = np.empty_like(inventory)
    beta = params.beta
    for asset in range(params.M):
        lam, gamma, rho, phi, psi = (getattr(params, name)[asset, asset]
                                     for name in ("Lambda", "Gamma", "R", "Phi", "Psi"))
        inverse = 1.0 / (beta * lam)
        drift = np.array([
            [0, -inverse, -inverse],
            [-phi, 0, 0],
            [0, -params.varpi * gamma * inverse, -rho - params.varpi * gamma * inverse],
        ])
        loading = np.array([-params.chi_g / beta, 0,
                             gamma * (1 - params.varpi * params.chi_g / beta)])
        flow = params.scenario_orders[:, asset] / params.twap_tau
        state = _exact_scalar_paths(params, drift, loading, flow, psi, times, selected)
        if state is None:
            state = _stable_scalar_paths(params, drift, loading, flow, psi, times, selected)
        inventory[:, :, asset] = state[:, :, 0]
        impact[:, :, asset] = state[:, :, 2]
    if not np.all(np.isfinite(inventory)) or not np.all(np.isfinite(impact)):
        raise RuntimeError("Mean-field boundary-value solve returned nonfinite values.")
    return {k: {"times": times, "mean_inventory": inventory[j], "impact": impact[j]}
            for j, k in enumerate(scenario_numbers)}
