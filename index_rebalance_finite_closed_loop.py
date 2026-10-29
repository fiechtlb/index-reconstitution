"""Closed-loop coefficients and paths used by the manuscript."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from _index_rebalance_common import (
    Array, FiniteGameParameters, _interp_grid, _sym,
)


def _observation_times(params: FiniteGameParameters, times: Array) -> Array:
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > params.T):
        raise ValueError("times must be increasing, finite, and contained in [0,T].")
    return times


@dataclass
class _DensePath:
    boundaries: Array
    interpolants: list[Callable]
    shape: tuple[int, ...]
    symmetric: bool = False

    def __call__(self, t: float) -> Array:
        t = float(np.clip(t, self.boundaries[0], self.boundaries[-1]))
        index = min(np.searchsorted(self.boundaries, t, side="right") - 1,
                    len(self.interpolants) - 1)
        value = self.interpolants[index](t).reshape(self.shape)
        return _sym(value) if self.symmetric else value

    def sample(self, times: Array) -> Array:
        return np.array([self(t) for t in times])


def _solve_dense(rhs, start, stop, initial, *, rtol, atol):
    """Keep adaptive dense output independent of the observation grid."""
    result = solve_ivp(rhs, (start, stop), np.asarray(initial).reshape(-1),
                       method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    if not result.success:
        raise RuntimeError(result.message)
    return result


@dataclass(frozen=True)
class FiniteMatrices:
    A: Array
    B_u: Array
    B_opp: Array
    B_g: Array
    E_I: Array
    Q: Array
    D: Array
    J: Array
    a: float
    d: float


def finite_matrices(params: FiniteGameParameters) -> FiniteMatrices:
    M = params.M
    I = np.eye(M)
    Z = np.zeros((M, M))
    N = float(params.N)
    impact_flow_weight = params.varpi_n
    A = np.block([[Z, Z, Z], [Z, Z, Z], [Z, Z, -params.R]])
    B_u = np.vstack(((N - 1.0) / N * I, (1.0 / N) * I, impact_flow_weight * params.Gamma))
    B_opp = np.vstack((-1.0 / N * I, (1.0 / N) * I, impact_flow_weight * params.Gamma))
    B_g = np.vstack((Z, Z, params.Gamma))
    E_I = np.hstack((Z, Z, I))
    Q = np.block([[params.Phi, params.Phi, Z], [params.Phi, params.Phi, Z], [Z, Z, Z]])
    C = np.block([[Z, Z, Z], [Z, N * I, Z], [Z, Z, N * I]])
    a = 2.0 - params.vartheta_n
    d = 2.0 + params.vartheta_n * (params.N - 1.0)
    D = (1.0 / a) * (np.eye(3 * M) - (params.vartheta_n / d) * C)
    J = (1.0 / a) * (np.eye(3 * M) - (2.0 / d) * C)
    return FiniteMatrices(A=A, B_u=B_u, B_opp=B_opp, B_g=B_g, E_I=E_I, Q=Q, D=D, J=J, a=a, d=d)


def terminal_reduced_P(params: FiniteGameParameters) -> Array:
    M = params.M
    Z = np.zeros((M, M))
    return np.block([[params.Psi, params.Psi, Z], [params.Psi, params.Psi, Z], [Z, Z, Z]])


def finite_theta(P: Array, mats: FiniteMatrices) -> Array:
    return mats.E_I + mats.B_u.T @ P


def finite_M_matrix(P: Array, params: FiniteGameParameters, mats: FiniteMatrices) -> Array:
    return mats.A + mats.B_opp @ np.linalg.solve(params.Lambda, finite_theta(P, mats) @ mats.J)


def solve_finite_riccati(
    params: FiniteGameParameters,
    times: Array,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Array:
    times = _observation_times(params, times)
    return _finite_riccati_path(params, rtol=rtol, atol=atol).sample(times)


def _finite_riccati_path(params: FiniteGameParameters, *, rtol, atol) -> _DensePath:
    mats = finite_matrices(params)
    inv_lam = np.linalg.inv(params.Lambda)

    def rhs(_t: float, flat: Array) -> Array:
        P = _sym(flat.reshape(3 * params.M, 3 * params.M))
        theta = finite_theta(P, mats)
        M_cl = finite_M_matrix(P, params, mats)
        dP = -(
            mats.Q
            + M_cl.T @ P
            + P @ M_cl
            - 2.0 * mats.D.T @ theta.T @ inv_lam @ theta @ mats.D
        )
        return _sym(dP).reshape(-1)

    result = _solve_dense(rhs, params.T, 0.0, terminal_reduced_P(params),
                          rtol=rtol, atol=atol)
    return _DensePath(np.array([0., params.T]), [result.sol],
                      (3 * params.M, 3 * params.M), symmetric=True)


def _post_common_rhs(
    t: float,
    state: Array,
    params: FiniteGameParameters,
    mats: FiniteMatrices,
    P_at: Callable,
    g: Array,
    with_alpha: bool,
) -> Array:
    dim = 3 * params.M
    ell = state[:dim]
    P = P_at(t)
    theta = finite_theta(P, mats)
    M_cl = finite_M_matrix(P, params, mats)
    eta = mats.B_u.T @ ell + params.chi_g * params.Lambda @ g
    xi = eta / mats.d
    zeta = -(params.N - 1.0) * eta / mats.d
    drift = mats.B_g @ g + mats.B_opp @ np.linalg.solve(params.Lambda, zeta)
    ell_dot = -(M_cl.T @ ell + P @ drift - 2.0 * mats.D.T @ theta.T @ np.linalg.solve(params.Lambda, xi))
    if not with_alpha:
        return ell_dot
    alpha_dot = -(float(drift.T @ ell) - float(xi.T @ np.linalg.solve(params.Lambda, xi)))
    return np.concatenate([ell_dot, [alpha_dot]])


def _pre_full_rhs(
    t: float,
    state: Array,
    params: FiniteGameParameters,
    mats: FiniteMatrices,
    P_at: Callable,
    with_alpha: bool,
) -> Array:
    dim = 3 * params.M
    ell_flat = state[: params.N * dim]
    ell = ell_flat.reshape(params.N, dim)
    P = P_at(t)
    theta = finite_theta(P, mats)
    M_cl = finite_M_matrix(P, params, mats)
    eta = ell @ mats.B_u
    eta_sum = eta.sum(axis=0)
    xi = eta / mats.a - params.vartheta_n * eta_sum[None, :] / (mats.a * mats.d)
    zeta = eta / mats.a - 2.0 * eta_sum[None, :] / (mats.a * mats.d)
    solved_xi = np.linalg.solve(params.Lambda, xi.T).T
    solved_zeta = np.linalg.solve(params.Lambda, zeta.T).T
    drift = solved_zeta @ mats.B_opp.T
    ell_dot = -(ell @ M_cl + drift @ P.T - solved_xi @ (2.0 * theta @ mats.D))
    if not with_alpha:
        return ell_dot.reshape(-1)
    alpha_dot = -(
        np.einsum("nd,nd->n", drift, ell) - np.einsum("nm,nm->n", xi, solved_xi)
    )
    return np.concatenate([ell_dot.reshape(-1), alpha_dot])


@dataclass
class FiniteSolution:
    params: FiniteGameParameters
    times: Array
    pre_times: Array
    post_times: Array
    P: Array
    ell_pre: Array
    alpha_pre: Array | None
    ell_post: Array
    alpha_post: Array | None
    _P_path: _DensePath | None = field(default=None, repr=False)
    _pre_path: _DensePath | None = field(default=None, repr=False)
    _post_paths: list[_DensePath] | None = field(default=None, repr=False)

    def P_at(self, t: float) -> Array:
        if self._P_path is not None:
            return self._P_path(t)
        return _interp_grid(self.times, self.P, t)

    def pre_ell_at(self, t: float) -> Array:
        if self._pre_path is not None:
            shape = self.ell_pre.shape[1:]
            return self._pre_path(t)[:int(np.prod(shape))].reshape(shape)
        return _interp_grid(self.pre_times, self.ell_pre, t)

    def post_ell_at(self, scenario: int, t: float) -> Array:
        if self._post_paths is not None:
            return self._post_paths[scenario - 1](t)[:3 * self.params.M]
        return _interp_grid(self.post_times, self.ell_post[scenario - 1], t)


def _solve_coefficients(
    params: FiniteGameParameters,
    times: Array,
    *,
    solve_alpha: bool,
    common_only: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> FiniteSolution:
    # Sample arrays remain public. Dense paths drive the ODEs, including the
    # terminal Riccati layer, and supply announcement values on arbitrary grids.
    times = _observation_times(params, times)
    mats = finite_matrices(params)
    P_path = _finite_riccati_path(params, rtol=rtol, atol=atol)
    P_values = P_path.sample(times)
    pre_times = np.unique(np.r_[0., times[times < params.T_ann], params.T_ann])
    post_times = np.unique(np.r_[params.T_ann, times[times > params.T_ann], params.T])
    dim = 3 * params.M

    ell_post = np.zeros((params.K, len(post_times), dim))
    alpha_post = np.zeros((params.K, len(post_times))) if solve_alpha else None
    post_paths = []
    announcement_values = []
    for scenario in range(1, params.K + 1):
        state = np.zeros(dim + (1 if solve_alpha else 0))
        boundaries = np.array([params.T_ann, params.T_twap, params.T_eff, params.T])
        interpolants = []
        for phase in reversed(range(len(boundaries) - 1)):
            left, right = boundaries[phase:phase + 2]
            g = (params.scenario_orders[scenario - 1] / params.twap_tau
                 if phase == 1 else np.zeros(params.M))
            result = _solve_dense(
                lambda t, y: _post_common_rhs(t, y, params, mats, P_path, g, solve_alpha),
                right, left, state, rtol=rtol, atol=atol)
            interpolants.insert(0, result.sol)
            state = result.y[:, -1]
        path = _DensePath(boundaries, interpolants, state.shape)
        post_paths.append(path)
        announcement_values.append(state)
        values = path.sample(post_times)
        ell_post[scenario - 1] = values[:, :dim]
        if solve_alpha:
            alpha_post[scenario - 1] = values[:, dim]

    priors = params.priors.mean(axis=0, keepdims=True) if common_only else params.priors
    announcement_values = np.asarray(announcement_values)
    ell_terminal = priors @ announcement_values[:, :dim]
    if solve_alpha:
        alpha_terminal = priors @ announcement_values[:, dim]
        terminal_pre = np.concatenate([ell_terminal.reshape(-1), alpha_terminal])
    else:
        terminal_pre = ell_terminal.reshape(-1)

    def pre_rhs(t: float, y: Array) -> Array:
        if common_only:
            return _post_common_rhs(t, y, params, mats, P_path, np.zeros(params.M), solve_alpha)
        return _pre_full_rhs(t, y, params, mats, P_path, solve_alpha)

    result = _solve_dense(pre_rhs, params.T_ann, 0., terminal_pre, rtol=rtol, atol=atol)
    pre_path = _DensePath(np.array([0., params.T_ann]), [result.sol], terminal_pre.shape)
    pre_values = pre_path.sample(pre_times)
    n_priors = len(priors)
    ell_pre = pre_values[:, : n_priors * dim].reshape(len(pre_times), n_priors, dim)
    alpha_pre = pre_values[:, n_priors * dim :] if solve_alpha else None

    return FiniteSolution(
        params=params,
        times=times,
        pre_times=pre_times,
        post_times=post_times,
        P=P_values,
        ell_pre=ell_pre,
        alpha_pre=alpha_pre,
        ell_post=ell_post,
        alpha_post=alpha_post,
        _P_path=P_path,
        _pre_path=pre_path,
        _post_paths=post_paths,
    )


def solve_finite_game(
    params: FiniteGameParameters,
    times: Array,
    *,
    solve_alpha: bool = True,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> FiniteSolution:
    """Return coefficient samples and dense evaluations on the full horizon.

    Observation times may omit events. The pre/post sample grids additionally
    contain their interval endpoints for exact announcement and terminal data.
    """
    return _solve_coefficients(params, times, solve_alpha=solve_alpha, rtol=rtol, atol=atol)


def solve_common_inventory_paths(
    *,
    params: FiniteGameParameters,
    times: Array,
    scenario_numbers: list[int],
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[int, dict[str, Array]]:
    """Sample selected common-state paths from zero inventories and impact."""

    times = _observation_times(params, times)
    solution = _solve_coefficients(params, times, solve_alpha=False, common_only=True,
                                   rtol=rtol, atol=atol)
    mats = finite_matrices(params)
    inv_lambda = np.linalg.inv(params.Lambda)

    paths: dict[int, dict[str, Array]] = {}
    for scenario_number in scenario_numbers:
        if not (1 <= scenario_number <= params.K):
            raise ValueError("scenario number must be between 1 and K.")

        def common_control(t: float, state: Array, *, post: bool, g: Array) -> Array:
            x_bar = state[: params.M]
            impact = state[params.M :]
            P = solution.P_at(t)
            theta = finite_theta(P, mats)
            ell = (solution.post_ell_at(scenario_number, t) if post
                   else solution.pre_ell_at(t)[0])
            eta = mats.B_u.T @ ell + params.chi_g * params.Lambda @ g
            z_common = np.concatenate([np.zeros(params.M), x_bar, impact])
            h = theta @ z_common + eta
            return -(1.0 / mats.d) * inv_lambda @ h

        common_path = np.empty((len(times), 2 * params.M))
        state = np.zeros(2 * params.M)
        boundaries = (0., params.T_ann, params.T_twap, params.T_eff, params.T)
        for phase, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            g = (params.scenario_orders[scenario_number - 1] / params.twap_tau
                 if phase == 2 else np.zeros(params.M))

            def rhs(t: float, state: Array) -> Array:
                mean_control = common_control(t, state, post=phase > 0, g=g)
                impact = state[params.M:]
                impact_dot = (-params.R @ impact
                              + params.Gamma @ (params.varpi_n * params.N * mean_control + g))
                return np.concatenate([mean_control, impact_dot])

            result = _solve_dense(rhs, left, right, state, rtol=rtol, atol=atol)
            indices = np.flatnonzero((times >= left) & (times <= right))
            if len(indices):
                common_path[indices] = result.sol(times[indices]).T
            state = result.y[:, -1]
        mean_inventory = common_path[:, : params.M]
        paths[scenario_number] = {
            "times": times,
            "mean_inventory": mean_inventory,
            "impact": common_path[:, params.M :],
        }
    return paths


def simulate_finite_scenarios(
    solution: FiniteSolution,
    times: np.ndarray,
) -> list[dict[str, Array | Callable | int]]:
    """Sample all scenarios and expose dense state/control evaluators.

    Dense evaluators return read-only arrays and share the most recent batch
    evaluation across scenarios, so quadrature need not use the observation grid.
    """

    params = solution.params
    times = _observation_times(params, times)
    mats = finite_matrices(params)
    inv_lambda_t = np.linalg.inv(params.Lambda).T
    n_scenarios = params.K

    def feedback_batch(
        t: float,
        inventories: np.ndarray,
        impact: np.ndarray,
        *,
        post_announcement: bool,
        indexer_flow: np.ndarray,
    ) -> np.ndarray:
        P = solution.P_at(t)
        theta = finite_theta(P, mats)
        mean_inventory = inventories.mean(axis=1)
        if post_announcement:
            ell = np.asarray(
                [
                    solution.post_ell_at(scenario, t)
                    for scenario in range(1, params.K + 1)
                ]
            )
            eta = ell @ mats.B_u + params.chi_g * indexer_flow @ params.Lambda.T
            xis = np.repeat((eta / mats.d)[:, None, :], params.N, axis=1)
        else:
            ell = solution.pre_ell_at(t)
            eta = ell @ mats.B_u
            eta_sum = eta.sum(axis=0)
            xi = eta / mats.a - params.vartheta_n * eta_sum[None, :] / (mats.a * mats.d)
            xis = np.repeat(xi[None, :, :], n_scenarios, axis=0)

        reduced_state = np.concatenate(
            [
                inventories - mean_inventory[:, None, :],
                np.repeat(mean_inventory[:, None, :], params.N, axis=1),
                np.repeat(impact[:, None, :], params.N, axis=1),
            ],
            axis=2,
        )
        rhs = reduced_state @ (theta @ mats.D).T + xis
        return -rhs @ inv_lambda_t

    def derivatives(
        t: float,
        inventories: np.ndarray,
        impact: np.ndarray,
        *,
        post_announcement: bool,
        indexer_flow: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        controls = feedback_batch(
            t,
            inventories,
            impact,
            post_announcement=post_announcement,
            indexer_flow=indexer_flow,
        )
        opportunist_flow = params.varpi_n * controls.sum(axis=1)
        impact_derivative = (
            -impact @ params.R.T + (opportunist_flow + indexer_flow) @ params.Gamma.T
        )
        return controls, impact_derivative

    inventory_shape = (n_scenarios, params.N, params.M)
    impact_shape = (n_scenarios, params.M)
    inventory_size = int(np.prod(inventory_shape))
    state = np.zeros(inventory_size + int(np.prod(impact_shape)))
    inventory_path = np.zeros((len(times),) + inventory_shape)
    impact_path = np.zeros((len(times),) + impact_shape)
    zero_flow = np.zeros((n_scenarios, params.M))
    scenario_flows = params.scenario_orders / params.twap_tau

    phases = [
        (0., params.T_ann, False, zero_flow),
        (params.T_ann, params.T_twap, True, zero_flow),
        (params.T_twap, params.T_eff, True, scenario_flows),
        (params.T_eff, params.T, True, zero_flow),
    ]
    forward_interpolants = []
    for left, right, post_announcement, indexer_flow in phases:
        indices = np.flatnonzero((times >= left) & (times <= right))
        phase_times = times[indices]

        def rhs(t: float, flat_state: np.ndarray) -> np.ndarray:
            inventories = flat_state[:inventory_size].reshape(inventory_shape)
            impact = flat_state[inventory_size:].reshape(impact_shape)
            inventory_derivative, impact_derivative = derivatives(
                t,
                inventories,
                impact,
                post_announcement=post_announcement,
                indexer_flow=indexer_flow,
            )
            return np.concatenate(
                [inventory_derivative.reshape(-1), impact_derivative.reshape(-1)]
            )

        result = _solve_dense(rhs, left, right, state, rtol=1e-7, atol=1e-9)
        forward_interpolants.append(result.sol)
        if len(indices):
            phase_states = result.sol(phase_times).T
            inventory_path[indices] = phase_states[:, :inventory_size].reshape(
                (len(indices),) + inventory_shape
            )
            impact_path[indices] = phase_states[:, inventory_size:].reshape(
                (len(indices),) + impact_shape
            )
        state = result.y[:, -1]

    last_time = None
    last_evaluation = None

    def evaluate(t: float):
        nonlocal last_time, last_evaluation
        t = float(t)
        if not np.isfinite(t) or not 0. <= t <= params.T:
            raise ValueError("evaluation time must be finite and contained in [0,T].")
        if last_time != t:
            phase = np.searchsorted([params.T_ann, params.T_twap, params.T_eff], t,
                                    side="right")
            flat = forward_interpolants[phase](t)
            inventories = flat[:inventory_size].reshape(inventory_shape)
            impact = flat[inventory_size:].reshape(impact_shape)
            flow = scenario_flows if params.T_twap <= t <= params.T_eff else zero_flow
            controls = feedback_batch(t, inventories, impact,
                                      post_announcement=t >= params.T_ann, indexer_flow=flow)
            for value in (inventories, impact, controls):
                value.setflags(write=False)
            last_time = t
            last_evaluation = inventories, impact, controls
        return last_evaluation

    control_path = np.zeros_like(inventory_path)
    indexer_flow_path = np.zeros_like(impact_path)
    for j, t in enumerate(times):
        if params.T_twap <= t <= params.T_eff:
            indexer_flow = scenario_flows
        else:
            indexer_flow = zero_flow
        control_path[j] = evaluate(t)[2]
        indexer_flow_path[j] = indexer_flow

    return [
        {
            "scenario": scenario + 1,
            "times": times,
            "X": inventory_path[:, scenario],
            "mean_inventory": inventory_path[:, scenario].mean(axis=1),
            "impact": impact_path[:, scenario],
            "controls": control_path[:, scenario],
            "indexer_flow": indexer_flow_path[:, scenario],
            "X_at": lambda t, k=scenario: evaluate(t)[0][k],
            "impact_at": lambda t, k=scenario: evaluate(t)[1][k],
            "controls_at": lambda t, k=scenario: evaluate(t)[2][k],
        }
        for scenario in range(n_scenarios)
    ]
