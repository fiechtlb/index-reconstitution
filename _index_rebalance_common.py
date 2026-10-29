from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray


def _sym(x: Array) -> Array:
    """Return the symmetric part of a square matrix."""

    return 0.5 * (x + x.T)


def _as_array(x: Array, *, ndim: int, name: str) -> Array:
    value = np.asarray(x, dtype=float)
    if value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {value.ndim}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_symmetric_matrix(matrix: Array, *, name: str, positive: bool = False) -> None:
    """Allow roundoff in symmetry/PSD, including genuinely zero penalties."""

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    scale = max(float(np.linalg.norm(matrix, ord=np.inf)), np.finfo(float).tiny)
    tolerance = 100 * np.finfo(float).eps * scale
    if np.linalg.norm(matrix - matrix.T, ord=np.inf) > tolerance:
        raise ValueError(f"{name} must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(_sym(matrix))
    if positive:
        if eigenvalues[0] <= 0:
            raise ValueError(f"{name} must be positive definite.")
    elif eigenvalues[0] < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite.")


def _normalize_parameter_arrays(params: object) -> None:
    for name in ("Lambda", "R", "Gamma", "Phi", "Psi", "scenario_orders"):
        object.__setattr__(params, name, _as_array(getattr(params, name), ndim=2, name=name))


def _validate_priors(priors: Array, *, name: str) -> None:
    if not np.all(np.isfinite(priors)) or np.any(priors < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    if not np.allclose(priors.sum(axis=-1), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Each probability vector in {name} must sum to one.")


def _interp_grid(times: Array, values: Array, t: float) -> Array:
    """Linearly interpolate an array whose first axis is time."""

    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    right = int(np.searchsorted(times, t, side="right"))
    left = right - 1
    weight = (t - times[left]) / (times[right] - times[left])
    return (1.0 - weight) * values[left] + weight * values[right]


def make_time_grid(
    T: float,
    T_ann: float,
    T_twap: float,
    T_eff: float,
    *,
    n_pre: int = 80,
    n_announcement_to_twap: int = 60,
    n_twap: int = 40,
    n_post: int = 60,
) -> Array:
    """Build a grid containing every model event time exactly once."""

    parts = [
        np.linspace(0.0, T_ann, n_pre),
        np.linspace(T_ann, T_twap, n_announcement_to_twap),
        np.linspace(T_twap, T_eff, n_twap),
        np.linspace(T_eff, T, n_post),
    ]
    return np.unique(np.concatenate(parts))


def _solve_backward(
    rhs: Callable[[float, Array], Array],
    t0: float,
    t1: float,
    terminal_value: Array,
    eval_times: Array,
    *,
    rtol: float,
    atol: float,
    max_step: float | None = None,
) -> Array:
    eval_times = np.asarray(eval_times, dtype=float)
    terminal_value = np.asarray(terminal_value, dtype=float)
    if eval_times[0] < t0 - 1e-12 or eval_times[-1] > t1 + 1e-12:
        raise ValueError("eval_times must lie inside [t0,t1].")
    if max_step is None:
        max_step = max((t1 - t0) / max(len(eval_times) - 1, 1), 1e-4)
    sol = solve_ivp(
        rhs,
        (t1, t0),
        terminal_value.reshape(-1),
        t_eval=eval_times[::-1],
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:, ::-1].T.reshape((len(eval_times),) + terminal_value.shape)


def _solve_forward(
    rhs: Callable[[float, Array], Array],
    t0: float,
    t1: float,
    initial_value: Array,
    eval_times: Array,
    *,
    rtol: float,
    atol: float,
    max_step: float | None = None,
) -> Array:
    eval_times = np.asarray(eval_times, dtype=float)
    initial_value = np.asarray(initial_value, dtype=float)
    if eval_times[0] < t0 - 1e-12 or eval_times[-1] > t1 + 1e-12:
        raise ValueError("eval_times must lie inside [t0,t1].")
    if max_step is None:
        max_step = max((t1 - t0) / max(len(eval_times) - 1, 1), 1e-4)
    sol = solve_ivp(
        rhs,
        (t0, t1),
        initial_value.reshape(-1),
        t_eval=eval_times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T.reshape((len(eval_times),) + initial_value.shape)


def _validate_common_parameters(
    *,
    Lambda: Array,
    R: Array,
    Gamma: Array,
    Phi: Array,
    Psi: Array,
    T_ann: float,
    T_twap: float,
    T_eff: float,
    T: float,
    twap_tau: float,
    scenario_orders: Array,
) -> tuple[int, int]:
    if not np.all(np.isfinite([T_ann, T_twap, T_eff, T])):
        raise ValueError("Event times must be finite.")
    if not (0.0 < T_ann < T_twap < T_eff < T):
        raise ValueError("Need 0 < T_ann < T_twap < T_eff < T.")
    duration = T_eff - T_twap
    if (not np.isfinite(twap_tau) or twap_tau <= 0
            or not np.isclose(twap_tau, duration, rtol=1e-12,
                              atol=16 * np.finfo(float).eps * max(abs(T_twap), abs(T_eff)))):
        raise ValueError("twap_tau must equal T_eff - T_twap.")
    execution = _as_array(Lambda, ndim=2, name="Lambda")
    M = execution.shape[0]
    if M == 0:
        raise ValueError("Coefficient matrices must be nonempty.")
    for name, raw in (
        ("Lambda", Lambda),
        ("R", R),
        ("Gamma", Gamma),
        ("Phi", Phi),
        ("Psi", Psi),
    ):
        matrix = _as_array(raw, ndim=2, name=name)
        if matrix.shape != (M, M):
            raise ValueError(f"{name} must have shape {(M, M)}, got {matrix.shape}.")
    _validate_symmetric_matrix(execution, name="Lambda", positive=True)
    for name, matrix in (("Phi", Phi), ("Psi", Psi), ("Gamma", Gamma)):
        _validate_symmetric_matrix(matrix, name=name)
    # For K(t)=exp(-R t) Gamma, this covariance condition ensures the
    # symmetric time kernel is positive semidefinite. It permits nonsymmetric
    # R and noncommuting matrices, unlike the paper's simpler sufficient example.
    _validate_symmetric_matrix(R @ Gamma + Gamma @ R.T,
                               name="R Gamma + Gamma R.T")
    orders = _as_array(scenario_orders, ndim=2, name="scenario_orders")
    if orders.shape[0] == 0 or orders.shape[1] != M:
        raise ValueError("scenario_orders must have shape (K,M).")
    return M, int(orders.shape[0])


def _validate_loading(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_population_weight(value: float, *, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class FiniteGameParameters:
    N: int
    Lambda: Array
    R: Array
    Gamma: Array
    Phi: Array
    Psi: Array
    T_ann: float
    T_twap: float
    T_eff: float
    T: float
    twap_tau: float
    scenario_orders: Array
    priors: Array
    varpi_n: float = 1.0
    chi_u: float = 1.0
    chi_g: float = 1.0

    def __post_init__(self) -> None:
        if (isinstance(self.N, (bool, np.bool_))
                or not isinstance(self.N, (int, np.integer)) or self.N < 1):
            raise ValueError("N must be a positive integer.")
        _normalize_parameter_arrays(self)
        _validate_loading(self.chi_u, name="chi_u")
        _validate_loading(self.chi_g, name="chi_g")
        _M, K = _validate_common_parameters(
            Lambda=self.Lambda,
            R=self.R,
            Gamma=self.Gamma,
            Phi=self.Phi,
            Psi=self.Psi,
            T_ann=self.T_ann,
            T_twap=self.T_twap,
            T_eff=self.T_eff,
            T=self.T,
            twap_tau=self.twap_tau,
            scenario_orders=self.scenario_orders,
        )
        _validate_population_weight(self.varpi_n, name="varpi_n")
        if self.vartheta_n > 1.0:
            raise ValueError("chi_u * varpi_n must be at most 1.")
        priors = _as_array(self.priors, ndim=2, name="priors")
        if priors.shape != (self.N, K):
            raise ValueError("priors must have shape (N,K).")
        _validate_priors(priors, name="priors")
        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "twap_tau", self.T_eff - self.T_twap)

    @property
    def M(self) -> int:
        return int(self.Lambda.shape[0])

    @property
    def K(self) -> int:
        return int(self.scenario_orders.shape[0])

    @property
    def vartheta_n(self) -> float:
        return float(self.chi_u * self.varpi_n)

    def g(self, scenario: int, t: float) -> Array:
        if scenario == 0:
            return np.zeros(self.M)
        if not (1 <= scenario <= self.K):
            raise ValueError("scenario must be 0 or between 1 and K.")
        if self.T_twap <= float(t) <= self.T_eff:
            return self.scenario_orders[scenario - 1] / self.twap_tau
        return np.zeros(self.M)


@dataclass(frozen=True)
class MeanFieldParameters:
    """Primitives for the mean-field model under ``varpi_n=varpi/n``."""

    Lambda: Array
    R: Array
    Gamma: Array
    Phi: Array
    Psi: Array
    T_ann: float
    T_twap: float
    T_eff: float
    T: float
    twap_tau: float
    scenario_orders: Array
    average_prior: Array
    chi_u: float = 1.0
    chi_g: float = 1.0
    varpi: float = 1.0

    def __post_init__(self) -> None:
        _normalize_parameter_arrays(self)
        _validate_population_weight(self.varpi, name="varpi")
        _validate_loading(self.chi_u, name="chi_u")
        _validate_loading(self.chi_g, name="chi_g")
        _M, K = _validate_common_parameters(
            Lambda=self.Lambda,
            R=self.R,
            Gamma=self.Gamma,
            Phi=self.Phi,
            Psi=self.Psi,
            T_ann=self.T_ann,
            T_twap=self.T_twap,
            T_eff=self.T_eff,
            T=self.T,
            twap_tau=self.twap_tau,
            scenario_orders=self.scenario_orders,
        )
        prior = np.asarray(self.average_prior, dtype=float)
        if prior.shape != (K,):
            raise ValueError("average_prior must have shape (K,).")
        _validate_priors(prior, name="average_prior")
        object.__setattr__(self, "average_prior", prior)
        object.__setattr__(self, "twap_tau", self.T_eff - self.T_twap)

    @property
    def M(self) -> int:
        return int(self.Lambda.shape[0])

    @property
    def K(self) -> int:
        return int(self.scenario_orders.shape[0])

    @property
    def beta(self) -> float:
        return 2.0 + self.varpi * self.chi_u

    def g(self, scenario: int, t: float) -> Array:
        if scenario == 0:
            return np.zeros(self.M)
        if not (1 <= scenario <= self.K):
            raise ValueError("scenario must be 0 or between 1 and K.")
        if self.T_twap <= float(t) <= self.T_eff:
            return self.scenario_orders[scenario - 1] / self.twap_tau
        return np.zeros(self.M)
