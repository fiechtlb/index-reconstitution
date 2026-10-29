"""Calibration and metrics for the nine manuscript figures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm

from _index_rebalance_common import FiniteGameParameters, make_time_grid

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "paper" / "figures" / "finite_player_game"
DEFAULT_CACHE_ROOT = Path(gettempdir()) / "index-rebalance/cache/finite_player_game"
DENSE_N_VALUES = tuple(range(1, 101))
CANDIDATE_ASSETS = tuple(range(5))
PLAYER_INVENTORY_N = 20
CALIBRATION_VERSION = "five-asset-daily-v7-dense-events"
BASELINE_INITIAL_HOLDING_ADV = 7.0
IMPACT_YLABEL = "Impact (USD/share)"
PLAYER_INVENTORY_YLABEL = "Inventory (mn shares)"
SIGNED_TOTAL_INVENTORY_YLABEL = "Signed total inventory (mn shares)"


@dataclass(frozen=True)
class ExperimentCalibration:
    tickers: tuple[str, ...]
    prices: np.ndarray
    adv: np.ndarray
    gamma_diagonal: np.ndarray
    hourly_execution_diagonal: np.ndarray
    initial_assets: tuple[int, ...]
    candidate_assets: tuple[int, ...]
    scenario_scores: np.ndarray
    initial_holding_adv_fraction: float
    trading_hours_per_day: float
    resilience_half_life_days: float
    running_penalty_lambda_ratio: float
    terminal_penalty_lambda_ratio: float
    prior_concentration: float
    prior_seed: int
    float_proxy_adv_days: float = 100.0
    T_ann: float = 10.0
    T_twap: float = 19.0
    T_eff: float = 20.0
    T: float = 22.0
    twap_tau: float = 1.0
    chi_u: float = 1.0
    chi_g: float = 1.0

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def float_shares(self) -> np.ndarray:
        """Illustrative floating shares, in millions, from daily turnover."""

        return self.float_proxy_adv_days * self.adv


@dataclass(frozen=True)
class ScenarioSpecification:
    pairs: tuple[tuple[int, int], ...]
    orders: np.ndarray
    average_prior: np.ndarray
    initial_holdings: np.ndarray

@dataclass(frozen=True)
class ExperimentGrid:
    """Event-aware integration edges and a solve grid containing midpoints."""

    edges: np.ndarray
    times: np.ndarray


def baseline_calibration() -> ExperimentCalibration:
    """Return the paper's deterministic five-asset daily calibration."""

    return ExperimentCalibration(
        tickers=("INTC", "CSCO", "EBAY", "DOW", "COST"),
        prices=np.array([30.26, 24.21, 53.41, 49.48, 121.37]),
        adv=np.array([5.489503, 5.598870, 2.506048, 1.453332, 0.540008]),
        gamma_diagonal=np.array([1.09, 0.63, 4.50, 6.90, 25.30]),
        hourly_execution_diagonal=np.array([0.44, 0.244, 2.50, 3.01, 10.60]),
        initial_assets=(0, 1),
        candidate_assets=CANDIDATE_ASSETS,
        scenario_scores=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        initial_holding_adv_fraction=BASELINE_INITIAL_HOLDING_ADV,
        trading_hours_per_day=6.5,
        resilience_half_life_days=0.5,
        running_penalty_lambda_ratio=0.25,
        terminal_penalty_lambda_ratio=13_000.0,
        prior_concentration=50.0,
        prior_seed=1729,
    )


def build_scenarios(calibration: ExperimentCalibration) -> ScenarioSpecification:
    """Build all two-of-five membership scenarios and their average prior."""

    pairs = tuple(combinations(calibration.candidate_assets, 2))
    old_membership = np.zeros(calibration.n_assets)
    old_membership[list(calibration.initial_assets)] = 1.0
    initial_holdings = (
        calibration.initial_holding_adv_fraction
        * calibration.adv
        * old_membership
    )
    old_value = float(np.dot(calibration.prices, initial_holdings))
    float_shares = calibration.float_shares

    orders: list[np.ndarray] = []
    weights: list[float] = []
    for pair in pairs:
        membership = np.zeros(calibration.n_assets)
        membership[list(pair)] = 1.0
        post_scale = old_value / np.dot(
            calibration.prices,
            float_shares * membership,
        )
        post_holdings = post_scale * float_shares * membership
        orders.append(post_holdings - initial_holdings)

        first, second = pair
        weights.append(
            float(
                calibration.scenario_scores[first]
                * calibration.scenario_scores[second]
            )
        )

    average_prior = np.asarray(weights, dtype=float)
    average_prior /= average_prior.sum()
    return ScenarioSpecification(
        pairs=pairs,
        orders=np.asarray(orders),
        average_prior=average_prior,
        initial_holdings=initial_holdings,
    )


def calibrated_matrices(
    calibration: ExperimentCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return daily-clock ``Lambda, R, Gamma, Q, Q_T`` matrices."""

    lambda_diagonal = (
        calibration.hourly_execution_diagonal
        / calibration.trading_hours_per_day
    )
    resilience_diagonal = np.full(
        calibration.n_assets,
        np.log(2.0) / calibration.resilience_half_life_days,
    )
    running_penalty_diagonal = (
        calibration.running_penalty_lambda_ratio * lambda_diagonal
    )
    terminal_penalty_diagonal = (
        calibration.terminal_penalty_lambda_ratio * lambda_diagonal
    )
    return (
        np.diag(lambda_diagonal),
        np.diag(resilience_diagonal),
        np.diag(calibration.gamma_diagonal),
        np.diag(running_penalty_diagonal),
        np.diag(terminal_penalty_diagonal),
    )


def sample_nested_priors(
    average_prior: np.ndarray,
    n_players: int,
    *,
    concentration: float,
    seed: int,
) -> np.ndarray:
    """Draw a reproducible prefix of one common prior pool."""

    if n_players < 1:
        raise ValueError("n_players must be positive.")
    rng = np.random.default_rng(seed)
    return rng.dirichlet(concentration * average_prior, size=n_players)


def build_prior_pool(
    calibration: ExperimentCalibration,
    scenarios: ScenarioSpecification,
    *,
    max_players: int = max(DENSE_N_VALUES),
) -> np.ndarray:
    return sample_nested_priors(
        scenarios.average_prior,
        max_players,
        concentration=calibration.prior_concentration,
        seed=calibration.prior_seed,
    )


def make_experiment_grid() -> ExperimentGrid:
    """Include every event and every midpoint used by figure quadrature."""

    calibration = baseline_calibration()
    edges = make_time_grid(
        calibration.T, calibration.T_ann, calibration.T_twap, calibration.T_eff,
        n_pre=41, n_announcement_to_twap=61, n_twap=81, n_post=41,
    )
    times = np.unique(np.concatenate([edges, 0.5 * (edges[:-1] + edges[1:])]))
    return ExperimentGrid(edges=edges, times=times)


def make_finite_game_parameters(
    *,
    calibration: ExperimentCalibration,
    scenarios: ScenarioSpecification,
    matrices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    priors: np.ndarray,
    varpi_n: float = 1.0,
) -> FiniteGameParameters:
    """Translate the experiment calibration into the finite-solver API."""

    n_players = int(priors.shape[0])
    Lambda, R, Gamma, Phi, Psi = matrices
    return FiniteGameParameters(
        N=n_players,
        Lambda=Lambda,
        R=R,
        Gamma=Gamma,
        Phi=Phi,
        Psi=Psi,
        T_ann=calibration.T_ann,
        T_twap=calibration.T_twap,
        T_eff=calibration.T_eff,
        T=calibration.T,
        twap_tau=calibration.twap_tau,
        scenario_orders=scenarios.orders,
        priors=priors,
        varpi_n=varpi_n,
        chi_u=calibration.chi_u,
        chi_g=calibration.chi_g,
    )


def anchored_path(raw_path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a CLI path relative to the repository, never the process CWD."""

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def cache_signature(
    *,
    calibration: ExperimentCalibration,
    scenarios: ScenarioSpecification,
    matrices: tuple[np.ndarray, ...],
    grid: ExperimentGrid,
    n_players: int,
) -> str:
    """Invalidate cached paths whenever a calibration or solver input changes."""

    payload = dict(vars(calibration), version=CALIBRATION_VERSION, n_players=n_players)
    arrays = [payload.pop(key) for key, value in tuple(payload.items())
              if isinstance(value, np.ndarray)]
    arrays.extend((scenarios.orders, scenarios.average_prior, grid.edges, grid.times, *matrices))
    hasher = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode())
        hasher.update(str(contiguous.shape).encode())
        hasher.update(contiguous.view(np.uint8))
    return hasher.hexdigest()


def asset_label(asset: int) -> str:
    return f"Asset {asset + 1}"


@dataclass
class EquilibriumSummary:
    """Only the arrays consumed by manuscript figures."""

    n_players: int
    times: np.ndarray
    mean_inventory: np.ndarray
    impact: np.ndarray
    indexer_total_cost: np.ndarray
    sample_average_prior: np.ndarray
    player_candidate_inventory: np.ndarray | None = None

    @property
    def contains_player_paths(self) -> bool:
        return self.player_candidate_inventory is not None


def midpoint_indices(
    times: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate every quadrature midpoint on an event-aware solve grid."""

    times = np.asarray(times, dtype=float)
    edges = np.asarray(edges, dtype=float)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    right = np.clip(np.searchsorted(times, midpoints), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    use_left = np.abs(times[left] - midpoints) <= np.abs(times[right] - midpoints)
    indices = np.where(use_left, left, right)
    if not np.allclose(times[indices], midpoints, atol=1e-10, rtol=0.0):
        raise RuntimeError("The solve grid does not contain every quadrature midpoint.")
    return indices, midpoints, np.diff(edges)


def event_index(times: np.ndarray, event_time: float) -> int:
    matches = np.flatnonzero(
        np.isclose(times, event_time, atol=1e-11, rtol=0.0)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one grid point at event time {event_time}."
        )
    return int(matches[0])


def scenario_indexer_cost(
    *,
    params: FiniteGameParameters,
    path: Mapping[str, Any],
    edges: np.ndarray,
    scenario: int | None = None,
) -> float:
    """Indexer execution cost above fundamentals, using dense paths if available.

    Dense evaluation uses an explicit scenario index or the path's ``scenario``
    metadata to identify prescribed flow. Sample-only paths retain midpoint
    quadrature on ``edges``.
    """

    if scenario is None:
        scenario = path.get("scenario")
    if scenario is not None and _has_dense_paths([path]):
        if not 1 <= scenario <= params.K:
            raise ValueError("scenario must be between 1 and K.")
        flow = params.scenario_orders[scenario - 1] / params.twap_tau

        def payment(time):
            controls = path["controls_at"](time)
            displacement = path["impact_at"](time) + params.Lambda @ (
                params.varpi_n * controls.sum(axis=0) + flow)
            return flow @ displacement

        cost, _, info = quad_vec(payment, params.T_twap, params.T_eff,
                                 epsrel=1e-7, epsabs=1e-8, full_output=True)
        if not info.success:
            raise RuntimeError(f"Indexer cost quadrature failed: {info.message}")
        return float(cost)

    indices, midpoints, widths = midpoint_indices(path["times"], edges)
    twap = (midpoints >= params.T_twap) & (midpoints < params.T_eff)
    impact = np.asarray(path["impact"])[indices][twap]
    flow = np.asarray(path["indexer_flow"])[indices][twap]
    aggregate = np.asarray(path["controls"])[indices][twap].sum(axis=1)
    displacement = impact + (params.varpi_n * aggregate + flow) @ params.Lambda.T
    return float(np.sum(widths[twap] * np.einsum("ta,ta->t", flow, displacement)))


def summarize_scenario_paths(
    *,
    params: FiniteGameParameters,
    paths: Sequence[Mapping[str, Any]],
    edges: np.ndarray,
    candidate_assets: Sequence[int],
    retain_player_paths: bool = False,
) -> EquilibriumSummary:
    if len(paths) != params.K:
        raise ValueError(f"Expected {params.K} scenario paths, received {len(paths)}.")
    times = np.asarray(paths[0]["times"], dtype=float)
    if any(not np.array_equal(times, path["times"]) for path in paths[1:]):
        raise ValueError("Scenario paths use inconsistent time grids.")
    player_paths = (np.stack([path["X"][:, :, list(candidate_assets)] for path in paths])
                    if retain_player_paths else None)
    return EquilibriumSummary(
        n_players=params.N,
        times=times,
        mean_inventory=np.stack([path["mean_inventory"] for path in paths]),
        impact=np.stack([path["impact"] for path in paths]),
        indexer_total_cost=np.array([
            scenario_indexer_cost(params=params, path=path, edges=edges, scenario=k)
            for k, path in enumerate(paths, 1)
        ]),
        sample_average_prior=params.priors.mean(axis=0),
        player_candidate_inventory=player_paths,
    )


def expected_indexer_cost(
    summary: EquilibriumSummary,
    average_prior: np.ndarray,
) -> float:
    return float(average_prior @ summary.indexer_total_cost)


@dataclass(frozen=True)
class TerminalTradingWealthSummary:
    """Trading wealth before inventory penalties, per unit of trader mass.

    ``common_prior_player_wealth`` uses the same supplied scenario prior for
    every player. ``subjective_player_wealth`` uses each player's own prior.
    Their totals need not agree when beliefs differ. Multiply wealth statistics
    by ``varpi_n`` to convert them to physical trader-mass quantities.
    """

    scenario_player_wealth: np.ndarray
    common_prior_player_wealth: np.ndarray
    subjective_player_wealth: np.ndarray

    @property
    def common_prior_total(self) -> float:
        return float(self.common_prior_player_wealth.sum())

    @property
    def common_prior_per_player(self) -> float:
        return float(self.common_prior_player_wealth.mean())

    @property
    def subjective_total(self) -> float:
        return float(self.subjective_player_wealth.sum())

    @property
    def subjective_per_player(self) -> float:
        return float(self.subjective_player_wealth.mean())


@dataclass(frozen=True)
class OpportunistTradingStatistics:
    """Scenario gross volumes and pre-penalty wealth from the same quadrature.

    Gross volume has shape (number of dispersion fractions, number of scenarios).
    Wealth has shape (number of scenarios, number of players). The error is the
    adaptive integrator's joint max-norm estimate, not a formal error bound.
    """

    dispersion_fractions: np.ndarray
    scenario_gross_volume: np.ndarray
    scenario_player_wealth: np.ndarray
    estimated_quadrature_error: float | None


def _has_dense_paths(paths: Sequence[Mapping[str, Any]]) -> bool:
    return all(callable(path.get("controls_at")) and callable(path.get("impact_at"))
               for path in paths)


def integrate_opportunist_trading_statistics(
    *,
    params: FiniteGameParameters,
    paths: Sequence[Mapping[str, Any]],
    dispersion_fractions: Sequence[float] | None = None,
    edges: np.ndarray | None = None,
    epsrel: float = 1e-7,
    epsabs: float = 1e-8,
) -> OpportunistTradingStatistics:
    """Integrate gross volume and trading wealth, separating model event times.

    Dense ``controls_at`` and ``impact_at`` callbacks use adaptive quadrature on
    each continuous phase. Absolute control values are integrated before players
    or scenarios are averaged. Dispersion fractions scale each player's control
    deviation from the mean, preserving aggregate control. Wealth uses the
    original controls, before inventory penalties, under the illustrations'
    constant fundamental price and zero initial inventories/cash.
    An empty dispersion vector requests wealth alone.

    When dense callbacks are absent, ``edges`` is required and midpoint
    quadrature uses sampled paths. Its error must be assessed by grid refinement.
    All values use the model's per-unit-mass convention.
    """

    if len(paths) != params.K:
        raise ValueError(f"Expected {params.K} scenario paths, received {len(paths)}.")
    fractions = np.asarray([1.] if dispersion_fractions is None
                           else dispersion_fractions, dtype=float)
    if (fractions.ndim != 1 or not np.all(np.isfinite(fractions))
            or np.any((fractions < 0.) | (fractions > 1.))):
        raise ValueError("dispersion_fractions must be a vector in [0,1].")
    gross_size = len(fractions) * params.K

    def rates_and_wealth(controls, impact, flow):
        common = controls.mean(axis=1, keepdims=True)
        dispersed = (common[None] + fractions[:, None, None, None]
                     * (controls - common)[None])
        gross = np.abs(dispersed).sum(axis=(2, 3))
        opponents = controls.sum(axis=1, keepdims=True) - controls
        execution_flow = controls + params.vartheta_n * opponents + params.chi_g * flow[:, None]
        displacement = impact[:, None] + execution_flow @ params.Lambda.T
        wealth = -np.einsum("kia,kia->ki", controls, displacement)
        return np.r_[gross.ravel(), wealth.ravel()]

    if _has_dense_paths(paths):
        def integrand(time):
            controls = np.stack([path["controls_at"](time) for path in paths])
            impact = np.stack([path["impact_at"](time) for path in paths])
            flow = np.array([params.g(k, time) for k in range(1, params.K+1)])
            return rates_and_wealth(controls, impact, flow)

        result = np.zeros(gross_size + params.K*params.N)
        error = 0.
        boundaries = (0., params.T_ann, params.T_twap, params.T_eff, params.T)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            value, phase_error, info = quad_vec(integrand, left, right, epsrel=epsrel,
                epsabs=epsabs, norm="max", full_output=True)
            if not info.success:
                raise RuntimeError(f"Trading statistics quadrature failed: {info.message}")
            result += value
            error += phase_error
    else:
        if edges is None:
            raise ValueError("edges is required when dense path callbacks are unavailable.")
        indices = [midpoint_indices(path["times"], edges)[0] for path in paths]
        result = np.zeros(gross_size + params.K*params.N)
        for j, width in enumerate(np.diff(edges)):
            controls = np.stack([path["controls"][index[j]] for path, index in zip(paths, indices)])
            impact = np.stack([path["impact"][index[j]] for path, index in zip(paths, indices)])
            flow = np.stack([path["indexer_flow"][index[j]] for path, index in zip(paths, indices)])
            result += width * rates_and_wealth(controls, impact, flow)
        error = None
    return OpportunistTradingStatistics(
        dispersion_fractions=fractions,
        scenario_gross_volume=result[:gross_size].reshape(len(fractions), params.K),
        scenario_player_wealth=result[gross_size:].reshape(params.K, params.N),
        estimated_quadrature_error=error,
    )


def scenario_terminal_trading_wealth(
    *,
    params: FiniteGameParameters,
    path: Mapping[str, Any],
    edges: np.ndarray,
    scenario: int | None = None,
) -> np.ndarray:
    """Cash plus fundamental terminal inventory value for each opportunist.

    This uses the illustrations' constant fundamental price, zero initial cash,
    and zero initial inventories. Fundamental-price payments cancel terminal
    inventory value, leaving minus the trading payments above fundamentals.
    Inventory penalties are excluded. Values use the model's per-unit-mass
    convention and are USD millions under the manuscript calibration.

    A dense path uses adaptive quadrature separated at every model event. A
    sample-only path uses midpoint quadrature on ``edges`` and needs refinement
    to assess accuracy, especially for wealth near zero.
    """

    if _has_dense_paths([path]):
        if scenario is None:
            scenario = path.get("scenario")
        if scenario is not None:
            if not 1 <= scenario <= params.K:
                raise ValueError("scenario must be between 1 and K.")
            twap_flow = params.scenario_orders[scenario - 1] / params.twap_tau
        else:
            times = np.asarray(path.get("times", []))
            active = (times >= params.T_twap) & (times <= params.T_eff)
            if not np.any(active) or "indexer_flow" not in path:
                raise ValueError("A dense path needs scenario metadata or an explicit scenario.")
            twap_flow = np.asarray(path["indexer_flow"])[np.flatnonzero(active)[0]]

        def payment(time):
            controls = path["controls_at"](time)
            flow = twap_flow if params.T_twap <= time <= params.T_eff else np.zeros(params.M)
            opponents = controls.sum(axis=0)[None] - controls
            displacement = path["impact_at"](time)[None] + (
                controls + params.vartheta_n*opponents + params.chi_g*flow[None]) @ params.Lambda.T
            return -np.einsum("ia,ia->i", controls, displacement)

        result = np.zeros(params.N)
        boundaries = (0., params.T_ann, params.T_twap, params.T_eff, params.T)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            value, _, info = quad_vec(payment, left, right, epsrel=1e-7, epsabs=1e-8,
                                      norm="max", full_output=True)
            if not info.success:
                raise RuntimeError(f"Trading wealth quadrature failed: {info.message}")
            result += value
        return result

    indices, _, widths = midpoint_indices(path["times"], edges)
    controls = np.asarray(path["controls"])[indices]
    impact = np.asarray(path["impact"])[indices]
    flow = np.asarray(path["indexer_flow"])[indices]
    opponents = controls.sum(axis=1, keepdims=True) - controls
    execution_flow = (
        controls + params.vartheta_n * opponents + params.chi_g * flow[:, None, :]
    )
    displacement = impact[:, None, :] + execution_flow @ params.Lambda.T
    return -np.einsum("t,tia,tia->i", widths, controls, displacement)


def summarize_terminal_trading_wealth(
    *,
    params: FiniteGameParameters,
    paths: Sequence[Mapping[str, Any]],
    edges: np.ndarray,
    average_prior: np.ndarray,
) -> TerminalTradingWealthSummary:
    """Separate common-prior and individual-subjective wealth expectations.

    Pass the population mean prior for the common-prior wealth comparison in
    the paper. Passing the sampled mean prior instead gives a different common
    evaluation. Neither common-prior convention is the sum of individual
    subjective expectations when beliefs differ.
    """

    if len(paths) != params.K:
        raise ValueError(f"Expected {params.K} scenario paths, received {len(paths)}.")
    prior = np.asarray(average_prior, dtype=float)
    if (prior.shape != (params.K,) or not np.all(np.isfinite(prior))
            or np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0)):
        raise ValueError("average_prior must be a probability vector with shape (K,).")
    scenario_wealth = integrate_opportunist_trading_statistics(
        params=params, paths=paths, edges=edges, dispersion_fractions=()).scenario_player_wealth
    return TerminalTradingWealthSummary(
        scenario_player_wealth=scenario_wealth,
        common_prior_player_wealth=prior @ scenario_wealth,
        subjective_player_wealth=np.einsum("ik,ki->i", params.priors, scenario_wealth),
    )


def player_netted_timing_metrics(
    *,
    summary: EquilibriumSummary,
    average_prior: np.ndarray,
    candidate_assets: Sequence[int],
    T_ann: float,
    T_twap: float,
    T_eff: float,
) -> dict[str, float]:
    candidate_array = np.asarray(candidate_assets, dtype=int)

    def scenario_inventories(event_time: float) -> np.ndarray:
        index = event_index(summary.times, event_time)
        return (
            summary.n_players
            * summary.mean_inventory[:, index][:, candidate_array]
        )

    at_announcement = scenario_inventories(T_ann)
    at_twap = scenario_inventories(T_twap)
    at_effective = scenario_inventories(T_eff)
    return {
        "inventory_at_announcement": float(
            average_prior @ np.sum(np.abs(at_announcement), axis=1)
        ),
        "inventory_at_twap": float(
            average_prior @ np.sum(np.abs(at_twap), axis=1)
        ),
        "inventory_change_during_twap": float(
            average_prior @ np.sum(np.abs(at_effective - at_twap), axis=1)
        ),
    }


def impact_state_event_metrics(
    *,
    summary: EquilibriumSummary,
    average_prior: np.ndarray,
    candidate_assets: Sequence[int],
    event_times: Mapping[str, float],
) -> dict[str, float]:
    """Expected candidate-asset impact magnitude at requested event times."""

    candidate_array = np.asarray(candidate_assets, dtype=int)
    return {
        label: float(
            average_prior
            @ np.sum(
                np.abs(
                    summary.impact[
                        :, event_index(summary.times, event_time)
                    ][:, candidate_array]
                ),
                axis=1,
            )
        )
        for label, event_time in event_times.items()
    }


def expected_gross_indexer_order(
    *,
    orders: np.ndarray,
    average_prior: np.ndarray,
    assets: Sequence[int],
) -> float:
    asset_array = np.asarray(assets, dtype=int)
    return float(average_prior @ np.sum(np.abs(orders[:, asset_array]), axis=1))


def no_opportunist_indexer_cost(
    *,
    params: FiniteGameParameters,
    edges: np.ndarray | None = None,
    scenario: int,
) -> tuple[float, float, float]:
    """Exact indexer self-impact cost with no opportunistic traders.

    The matrix exponential jointly integrates impact and its trading payment,
    including dense or singular resilience matrices without diagonalization.
    Initial impact is zero, as in the illustrations. ``edges`` is retained for
    callers of the former quadrature API. The result is independent of the grid.
    """

    if not 1 <= scenario <= params.K:
        raise ValueError("scenario must be between 1 and K.")
    flow = params.scenario_orders[scenario - 1] / params.twap_tau
    duration = params.T_eff - params.T_twap
    # Coordinates are impact I, a constant 1, and cumulative transient payment.
    drift = np.zeros((params.M + 2, params.M + 2))
    drift[:params.M, :params.M] = -params.R
    drift[:params.M, params.M] = params.Gamma @ flow
    drift[-1, :params.M] = flow
    transient = float(expm(duration * drift)[-1, params.M])
    execution = float(duration * (flow @ params.Lambda @ flow))
    return transient + execution, transient, execution


def scenario_title(members: Sequence[int]) -> str:
    """Describe the new index by its asset numbers."""

    assets = " and ".join(str(asset) for asset in members)
    return f"New Index: Assets {assets}"
