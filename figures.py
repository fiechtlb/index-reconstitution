from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Sequence
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "index-rebalance/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
from index_rebalance_finite_closed_loop import (simulate_finite_scenarios, solve_finite_game,
                                                solve_common_inventory_paths)
from index_rebalance_mean_field import solve_diagonal_mean_field_paths
from _index_rebalance_common import FiniteGameParameters, MeanFieldParameters
from experiments import (
    CALIBRATION_VERSION,
    DEFAULT_CACHE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DENSE_N_VALUES,
    EquilibriumSummary,
    ExperimentCalibration,
    ExperimentGrid,
    IMPACT_YLABEL,
    PLAYER_INVENTORY_N,
    PLAYER_INVENTORY_YLABEL,
    SIGNED_TOTAL_INVENTORY_YLABEL,
    ScenarioSpecification,
    anchored_path,
    asset_label,
    baseline_calibration,
    build_prior_pool,
    build_scenarios,
    cache_signature,
    calibrated_matrices,
    expected_gross_indexer_order,
    expected_indexer_cost,
    impact_state_event_metrics,
    integrate_opportunist_trading_statistics,
    make_experiment_grid,
    make_finite_game_parameters,
    midpoint_indices,
    no_opportunist_indexer_cost,
    player_netted_timing_metrics,
    scenario_title,
    summarize_scenario_paths,
)

ASSET_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")
MANUSCRIPT_SCENARIOS = (1, 6, 2, 8)
FINITE_COMPARISON_RTOL = 1e-10
FINITE_COMPARISON_ATOL = 1e-12
VOLUME_EPSREL = 1e-6
VOLUME_EPSABS = 1e-8

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )

def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


class SummaryCache:
    array_fields = ("times", "mean_inventory", "impact", "indexer_total_cost",
                    "sample_average_prior")

    def __init__(self, root: Path, *, force: bool = False):
        self.root, self.force = root, force

    def path_for(self, *, n_players, signature):
        return self.root / f"n{n_players:03d}_{signature[:20]}.npz"

    def load(self, path, *, signature, require_player_paths):
        if self.force or not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            has_players = metadata["contains_player_paths"]
            if metadata["signature"] != signature or (require_player_paths and not has_players):
                return None
            return EquilibriumSummary(
                n_players=metadata["n_players"],
                **{key: np.asarray(data[key]) for key in self.array_fields},
                player_candidate_inventory=(np.asarray(data["player_candidate_inventory"])
                                            if has_players else None))

    def save(self, path, *, signature, summary):
        metadata = dict(signature=signature, n_players=summary.n_players,
                        contains_player_paths=summary.contains_player_paths)
        arrays = {key: getattr(summary, key) for key in self.array_fields}
        if summary.contains_player_paths:
            arrays["player_candidate_inventory"] = summary.player_candidate_inventory
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata)), **arrays)

class ExperimentEngine:
    """Own calibration state and serve cached equilibrium summaries."""

    def __init__(
        self,
        *,
        cache_root: Path,
        grid: ExperimentGrid,
        force: bool = False,
        calibration: ExperimentCalibration | None = None,
    ) -> None:
        self.calibration = (
            baseline_calibration() if calibration is None else calibration
        )
        self.scenarios = build_scenarios(self.calibration)
        self.matrices = calibrated_matrices(self.calibration)
        self.prior_pool = build_prior_pool(self.calibration, self.scenarios)
        self.grid = grid
        self.cache = SummaryCache(cache_root, force=force)
        self._memory: dict[int, EquilibriumSummary] = {}
        self._no_opportunist_cost: float | None = None

    def _validate_population_request(self, n_players: int) -> int:
        if (
            isinstance(n_players, (bool, np.bool_))
            or not isinstance(n_players, (int, np.integer))
            or not 1 <= n_players <= len(self.prior_pool)
        ):
            raise ValueError(
                f"n_players must be an integer from 1 through {len(self.prior_pool)} "
                "for the configured prior pool."
            )
        return int(n_players)

    def params(self, n_players: int):
        n_players = self._validate_population_request(n_players)
        return make_finite_game_parameters(
            calibration=self.calibration,
            scenarios=self.scenarios,
            matrices=self.matrices,
            priors=self.prior_pool[:n_players],
        )

    def summary(
        self,
        n_players: int,
        *,
        require_player_paths: bool = False,
    ) -> EquilibriumSummary:
        n_players = self._validate_population_request(n_players)
        key = n_players
        memory_summary = self._memory.get(key)
        if memory_summary is not None and (
            not require_player_paths or memory_summary.contains_player_paths
        ):
            return memory_summary

        signature = cache_signature(
            calibration=self.calibration,
            scenarios=self.scenarios,
            matrices=self.matrices,
            grid=self.grid,
            n_players=n_players,
        )
        cache_path = self.cache.path_for(
            n_players=n_players,
            signature=signature,
        )
        cached = self.cache.load(
            cache_path,
            signature=signature,
            require_player_paths=require_player_paths,
        )
        if cached is not None:
            print(
                f"Cache hit: n={n_players}",
                flush=True,
            )
            self._memory[key] = cached
            return cached

        print(
            f"Solving: n={n_players}",
            flush=True,
        )
        params = self.params(n_players)
        solution = solve_finite_game(
            params, self.grid.times, solve_alpha=False, rtol=1e-6, atol=1e-8)
        paths = simulate_finite_scenarios(solution, self.grid.times)

        summary = summarize_scenario_paths(
            params=params,
            paths=paths,
            edges=self.grid.edges,
            candidate_assets=self.calibration.candidate_assets,
            retain_player_paths=require_player_paths,
        )
        self.cache.save(cache_path, signature=signature, summary=summary)
        self._memory[key] = summary
        return summary

    def expected_no_opportunist_cost(self) -> float:
        if self._no_opportunist_cost is not None:
            return self._no_opportunist_cost
        params = self.params(1)
        scenario_costs = np.asarray(
            [
                no_opportunist_indexer_cost(
                    params=params,
                    edges=self.grid.edges,
                    scenario=scenario,
                )[0]
                for scenario in range(1, params.K + 1)
            ]
        )
        self._no_opportunist_cost = float(
            self.scenarios.average_prior @ scenario_costs
        )
        return self._no_opportunist_cost

def mark_event_times(ax: plt.Axes, calibration: ExperimentCalibration) -> None:
    for event_time in (
        calibration.T_ann,
        calibration.T_twap,
        calibration.T_eff,
    ):
        ax.axvline(
            event_time,
            color="0.55",
            linestyle=":",
            linewidth=0.75,
        )

TAG = "n020_varpi_1_chi_u_1_chi_g_1_closed_loop"
SCENARIO_TAG = "scenarios_01_06_02_08"


def scenario_overview_output_path(output_root: Path, quantity: str) -> Path:
    directory = (output_root / "impact_state" if quantity == "impact_state"
                 else output_root / "inventory" / "total_inventory")
    return directory / f"{quantity}_{SCENARIO_TAG}_{TAG}.pdf"


def player_inventory_output_path(output_root: Path) -> Path:
    return (output_root / "inventory" / "player_inventory" /
            f"player_inventory_asset_02_{SCENARIO_TAG}_{TAG}.pdf")


def plot_scenario_overview(
    *,
    summary: EquilibriumSummary,
    calibration: ExperimentCalibration,
    scenarios: ScenarioSpecification,
    quantity: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    values = (summary.impact if quantity == "impact_state"
              else summary.n_players * summary.mean_inventory)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2), sharex=True)
    for ax, scenario_number in zip(axes.flat, MANUSCRIPT_SCENARIOS):
        scenario_index = scenario_number - 1
        for position, asset in enumerate(calibration.candidate_assets):
            ax.plot(summary.times, values[scenario_index, :, asset],
                    color=ASSET_COLORS[position], linestyle="-", linewidth=1.15)
        mark_event_times(ax, calibration)
        ax.axhline(0.0, color="0.70", linewidth=0.65)
        ax.set_title(scenario_title(tuple(asset + 1 for asset in scenarios.pairs[scenario_index])),
                     loc="center", fontsize=9.0, pad=2)
        ax.grid(color="0.91", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (trading days)")
    handles = [Line2D([0], [0], color=ASSET_COLORS[position], linewidth=1.6)
               for position in range(len(calibration.candidate_assets))]
    legend = fig.legend(handles, [asset_label(asset) for asset in calibration.candidate_assets],
                        loc="lower center", bbox_to_anchor=(0.5, 0.006),
                        ncol=len(handles), frameon=False)
    fig.add_artist(legend)
    fig.suptitle(title, y=0.99)
    fig.supylabel(ylabel, x=0.05)
    fig.tight_layout(rect=(0.045, 0.04, 0.995, 1.0), h_pad=0.45, w_pad=0.65)
    save_pdf(fig, output)


def _selected_players_for_asset(
    *,
    priors: np.ndarray,
    scenario_orders: np.ndarray,
    average_prior: np.ndarray,
    asset: int,
    twap_tau: float,
) -> tuple[list[int], list[str]]:
    expected_flow = priors @ scenario_orders[:, asset] / twap_tau
    average_expected_flow = float(
        average_prior @ scenario_orders[:, asset] / twap_tau
    )
    order = np.argsort(expected_flow)
    closest = int(np.argmin(np.abs(expected_flow - average_expected_flow)))
    players = [int(order[0]), closest, int(order[-1])]
    roles = ["Lowest expected flow", "Closest to average", "Highest expected flow"]
    return players, roles

def plot_player_inventory(
    *,
    summary: EquilibriumSummary,
    calibration: ExperimentCalibration,
    scenarios: ScenarioSpecification,
    priors: np.ndarray,
    asset: int,
    output: Path,
) -> None:
    if summary.player_candidate_inventory is None:
        raise ValueError("Player inventory paths were not retained in the cache.")
    candidate_position = calibration.candidate_assets.index(asset)
    players, labels = _selected_players_for_asset(
        priors=priors,
        scenario_orders=scenarios.orders,
        average_prior=summary.sample_average_prior,
        asset=asset,
        twap_tau=calibration.twap_tau,
    )
    colors = ("#2166ac", "#555555", "#b2182b")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.2), sharex=True, squeeze=False)
    axes_flat = axes.reshape(-1)
    for ax, scenario_number in zip(axes_flat, MANUSCRIPT_SCENARIOS):
        for player, label, color in zip(players, labels, colors):
            ax.plot(
                summary.times,
                summary.player_candidate_inventory[
                    scenario_number - 1, :, player, candidate_position
                ],
                color=color,
                linewidth=1.2,
                label=label,
            )
        mark_event_times(ax, calibration)
        ax.axhline(0.0, color="0.70", linewidth=0.65)
        ax.set_title(
            scenario_title(
                tuple(asset + 1 for asset in scenarios.pairs[scenario_number - 1]),
            ),
            loc="center",
            fontsize=9.0,
            pad=2,
        )
        ax.grid(color="0.91", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time (trading days)")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    fig.suptitle(
        f"{asset_label(asset)} Inventories Under Different Beliefs",
        y=0.99,
    )
    fig.supylabel(PLAYER_INVENTORY_YLABEL, x=0.05)
    fig.tight_layout(
        rect=(0.045, 0.04, 0.995, 1.0),
        h_pad=0.45,
        w_pad=0.65,
    )
    save_pdf(fig, output)

INVENTORY_SERIES = (
    ("inventory_at_announcement", r"$\mathsf{L}_n^{\mathrm{ann}}$", "#1f77b4", "-"),
    ("inventory_at_twap", r"$\mathsf{L}_n^{\mathrm{TWAP}}$", "#d55e00", "--"),
    ("inventory_change_during_twap", r"$\mathsf{L}_n^{\mathrm{chg}}$", "#2ca02c", "-."),
)

IMPACT_SERIES = (
    ("impact_at_announcement", r"At $T_{\mathrm{ann}}$", "#1f77b4", "-"),
    ("impact_at_twap", r"At $T_{\mathrm{TWAP}}$", "#d55e00", "--"),
    ("impact_at_effective", r"At $T_{\mathrm{impl}}$", "#2ca02c", "-."),
    ("impact_at_terminal", r"At $T$", "#9467bd", ":"),
)

def sensitivity_grid_output_path(output_root: Path, quantity: str) -> Path:
    directory = (
        output_root / "inventory" / "total_inventory"
        if quantity == "inventory" else output_root / quantity
    )
    return directory / f"{quantity}_sensitivity_grid_{TAG}.pdf"

BASELINE_RHO = 2.0 * np.log(2.0)
PANEL_SPECS = (
    ('n', DENSE_N_VALUES, 20.0, '(a) Number of Opportunists', r'Number of opportunists $n$'),
    ('rho', BASELINE_RHO * np.logspace(-2.0, 1.0, 13), BASELINE_RHO,
     '(b) Impact Resilience', r'$\rho$ (per trading day)'),
)


def sensitivity_panels(engine):
    panels = []
    for kind, values, baseline, title, xlabel in PANEL_SPECS:
        rows = []
        for value in values:
            n = int(value) if kind == 'n' else 20
            current = engine
            if kind == 'rho' and not np.isclose(value, baseline, rtol=0, atol=1e-14):
                current = ExperimentEngine(cache_root=engine.cache.root, grid=engine.grid,
                                           force=engine.cache.force, calibration=engine.calibration)
                matrices = [matrix.copy() for matrix in engine.matrices]
                matrices[1] = value * np.eye(matrices[0].shape[0])
                current.matrices = tuple(matrices)
            summary = current.summary(n)
            prior = current.scenarios.average_prior
            c = current.calibration
            row = player_netted_timing_metrics(
                summary=summary, average_prior=prior, candidate_assets=c.candidate_assets,
                T_ann=c.T_ann, T_twap=c.T_twap, T_eff=c.T_eff)
            row.update(impact_state_event_metrics(
                summary=summary, average_prior=prior, candidate_assets=c.candidate_assets,
                event_times=dict(impact_at_announcement=c.T_ann, impact_at_twap=c.T_twap,
                                 impact_at_effective=c.T_eff, impact_at_terminal=c.T)))
            row['indexer_cost'] = expected_indexer_cost(summary, prior)
            row['no_opportunist_cost'] = current.expected_no_opportunist_cost()
            row['indexer_cost_savings'] = row['no_opportunist_cost'] - row['indexer_cost']
            rows.append(row)
        panels.append(dict(kind=kind, values=np.asarray(values), baseline=baseline,
                           title=title, xlabel=xlabel,
                           metrics={key: np.asarray([row[key] for row in rows]) for key in rows[0]}))
    return panels


def plot_sensitivity_grid(panels, quantity, expected_order, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3 if quantity == 'indexer_cost' else 3.05),
                             squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        x, metrics = panel['values'], panel['metrics']
        if quantity == 'indexer_cost':
            ax.plot(x, metrics['indexer_cost_savings'], label='Total savings', color='black',
                    linestyle='-', linewidth=1.8)
            ax.axhline(0, color='0.45', linestyle='--', linewidth=0.75)
            ax.set_ylabel('Expected cost savings (USD mn)')
        else:
            series = INVENTORY_SERIES if quantity == 'inventory' else IMPACT_SERIES
            for key, label, color, style in series:
                ax.plot(x, metrics[key], label=label, color=color, linestyle=style, linewidth=1.6)
            ax.set_ylabel('Aggregate inventory (mn shares)' if quantity == 'inventory'
                          else 'Impact magnitude (USD/share)')
            if quantity == 'inventory':
                ax.axhline(expected_order, color='0.35', linestyle=':', linewidth=1.2,
                           label='Expected gross indexer order')
            ax.set_ylim(bottom=0)
        ax.set_xscale('log')
        ax.set_xlim(float(x[0]), float(x[-1]))
        if panel['kind'] == 'n':
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
        ax.axvline(panel['baseline'], color='0.6', linestyle=':', linewidth=0.8)
        ax.set_xlabel(panel['xlabel'])
        ax.set_title(panel['title'], pad=7)
        ax.grid(axis='y', color='0.9', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    legend_space = 0
    if quantity != 'indexer_cost':
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 0))
        legend_space = 0.09
    fig.tight_layout(rect=(0, legend_space, 1, 1), h_pad=1.4, w_pad=1.8)
    save_pdf(fig, output)


def generate_sensitivity_figures(engine, output_root):
    panels = sensitivity_panels(engine)
    expected_order = expected_gross_indexer_order(
        orders=engine.scenarios.orders, average_prior=engine.scenarios.average_prior,
        assets=engine.calibration.candidate_assets)
    outputs = []
    for quantity in ('inventory', 'impact_state', 'indexer_cost'):
        output = sensitivity_grid_output_path(output_root, quantity)
        plot_sensitivity_grid(panels, quantity, expected_order, output)
        outputs.append(output)
    return outputs


def belief_cache_signature(engine, n, fractions):
    c = engine.calibration
    metadata = dict(version='closed-loop-belief-dispersion-v3-adaptive',
                    calibration_version=CALIBRATION_VERSION, n_players=int(n),
                    T_ann=c.T_ann, T_twap=c.T_twap, T_eff=c.T_eff, T=c.T,
                    twap_tau=c.twap_tau, chi_u=c.chi_u, chi_g=c.chi_g,
                    population_scaling="varpi_1",
                    rtol=1e-6, atol=1e-8,
                    volume_epsrel=VOLUME_EPSREL, volume_epsabs=VOLUME_EPSABS)
    hasher = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode('utf-8'))
    inputs = [('times', engine.grid.times), ('edges', engine.grid.edges),
              ('base_priors', engine.prior_pool[:n]), ('dispersion_fractions', fractions),
              ('scenario_orders', engine.scenarios.orders)]
    inputs.extend(zip(('Lambda', 'R', 'Gamma', 'Q', 'Q_T'), engine.matrices))
    for label, values in inputs:
        array = np.ascontiguousarray(np.asarray(values, dtype='<f8'))
        hasher.update(label.encode('utf-8'))
        hasher.update(json.dumps(array.shape).encode('ascii'))
        hasher.update(array.view(np.uint8))
    return hasher.hexdigest()


def belief_dispersion_result(engine):
    fractions = np.linspace(0, 1, 11)
    populations = np.asarray((2, 5, 10, 20, 50, 100))
    cells = []
    for n in populations:
        params = engine.params(int(n))
        signature = belief_cache_signature(engine, n, fractions)
        path = (engine.cache.root / 'belief_dispersion_sensitivity' / 'cells'
                / f'n{n:03d}_{signature[:24]}.npz')
        if path.exists() and not engine.cache.force:
            with np.load(path, allow_pickle=False) as data:
                cell = {key: data[key] for key in data.files}
        else:
            print(f'Solving belief dispersion: n={n}', flush=True)
            solution = solve_finite_game(params, engine.grid.times, solve_alpha=False, rtol=1e-6, atol=1e-8)
            paths = simulate_finite_scenarios(solution, engine.grid.times)
            weights = engine.prior_pool[:n].mean(axis=0)
            statistics = integrate_opportunist_trading_statistics(
                params=params, paths=paths, dispersion_fractions=fractions,
                edges=engine.grid.edges, epsrel=VOLUME_EPSREL, epsabs=VOLUME_EPSABS)
            gross = statistics.scenario_gross_volume @ weights
            priors = engine.prior_pool[:n]
            dispersion = np.sqrt(np.mean(np.sum((priors-weights)**2, axis=1)))
            cell = dict(belief_dispersion=fractions*dispersion, expected_gross_volume=gross,
                        quadrature_error=np.asarray(statistics.estimated_quadrature_error))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **cell)
        cells.append(cell)
    gross = np.column_stack([c['expected_gross_volume'] for c in cells])
    return SimpleNamespace(
        n_values=populations, dispersion_fractions=fractions,
        belief_dispersion=np.column_stack([c['belief_dispersion'] for c in cells]),
        expected_gross_volume=gross,
        incremental_gross_volume_per_player=(gross-gross[0])/populations)


def generate_dispersion_figure(engine, output_root):
    result = belief_dispersion_result(engine)
    output = belief_dispersion_volume_output_path(output_root)
    plot_belief_dispersion_volume(result=result, output=output)
    return output


def generate_manuscript_figures(engine, output_root):
    summary = engine.summary(PLAYER_INVENTORY_N, require_player_paths=True)
    outputs = []
    for quantity, title, ylabel in (
        ("signed_total_inventory", "Aggregate Inventory by Scenario", SIGNED_TOTAL_INVENTORY_YLABEL),
        ("impact_state", "Impact by Scenario", IMPACT_YLABEL)):
        output = scenario_overview_output_path(output_root, quantity)
        plot_scenario_overview(summary=summary, calibration=engine.calibration,
                               scenarios=engine.scenarios, quantity=quantity, title=title,
                               ylabel=ylabel, output=output)
        outputs.append(output)
    output = player_inventory_output_path(output_root)
    plot_player_inventory(summary=summary, calibration=engine.calibration, scenarios=engine.scenarios,
                          priors=engine.prior_pool[:PLAYER_INVENTORY_N], asset=1, output=output)
    outputs.append(output)
    outputs.extend(generate_sensitivity_figures(engine, output_root))
    outputs.append(generate_dispersion_figure(engine, output_root))
    return outputs


def _n_tag(values: Sequence[int]) -> str:
    return "_".join(f"{int(value):03d}" for value in values)

def belief_dispersion_volume_output_path(
    output_root: Path,
    *,
    selected_n_values: Sequence[int] = (2, 5, 10, 20, 50, 100),
) -> Path:
    return (
        output_root
        / "traded_volume"
        / "belief_dispersion_sensitivity"
        / (
            "opportunist_trading_volume_by_belief_dispersion_n"
            f"{_n_tag(selected_n_values)}_"
            "varpi_1_chi_u_1_chi_g_1_closed_loop.pdf"
        )
    )

def _n_styles(n_values: Sequence[int]) -> dict[int, tuple[str, str]]:
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(n_values)))
    markers = ("o", "s", "^", "D", "v", "P")
    return {
        int(n_players): (colors[index], markers[index % len(markers)])
        for index, n_players in enumerate(n_values)
    }

def plot_belief_dispersion_volume(
    *,
    result: SimpleNamespace,
    output: Path,
) -> None:
    """Plot total volume and its dispersion-driven increment per player."""

    styles = _n_styles(result.n_values)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    markevery = max(1, len(result.dispersion_fractions) // 5)
    incremental = result.incremental_gross_volume_per_player
    for column, n_players in enumerate(result.n_values):
        color, marker = styles[int(n_players)]
        common_options: dict[str, Any] = {
            "color": color,
            "marker": marker,
            "markevery": markevery,
            "markersize": 3.5,
            "linewidth": 1.55,
            "label": rf"$n={int(n_players)}$",
        }
        axes[0].plot(
            result.belief_dispersion[:, column],
            result.expected_gross_volume[:, column],
            **common_options,
        )
        axes[1].plot(
            result.belief_dispersion[:, column],
            incremental[:, column],
            **common_options,
        )

    axes[0].set_title("Total Gross Volume", pad=7)
    axes[0].set_ylabel("Expected cumulative gross volume (mn shares)")
    axes[1].set_title("Dispersion-Driven Volume per Opportunist", pad=7)
    axes[1].set_ylabel("Incremental gross volume per opportunist (mn shares)")
    for axis in axes:
        axis.set_xlabel(r"RMS belief dispersion $\mathcal{D}_{n,\delta}$")
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles = [
        Line2D(
            [0],
            [0],
            color=styles[int(n_players)][0],
            marker=styles[int(n_players)][1],
            markersize=4.0,
            linewidth=1.55,
            label=rf"$n={int(n_players)}$",
        )
        for n_players in result.n_values
    ]
    fig.suptitle(
        "Opportunist Trading Volume Under Mean-Preserving Belief Dispersion",
        y=0.985,
    )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=6,
        frameon=False,
        fontsize=8.0,
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.94), w_pad=1.6)
    save_pdf(fig, output)

def _candidate_panel_figure(
    *,
    times: np.ndarray,
    series: Sequence[tuple[str, np.ndarray, str, str, float]],
    calibration: ExperimentCalibration,
    candidate_assets: Sequence[int],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.8), sharex=True)
    axes_flat = np.asarray(axes).reshape(-1)
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for column, asset in enumerate(candidate_assets):
        ax = axes_flat[column]
        for label, values, color, linestyle, linewidth in series:
            ax.plot(
                times,
                values[:, column],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
            )
            if column == 0:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                    )
                )
                legend_labels.append(label)
        mark_event_times(ax, calibration)
        ax.axhline(0.0, color="0.65", linestyle="--", linewidth=0.65)
        ax.set_title(
            f"Asset {asset + 1} ({calibration.tickers[asset]})",
            loc="center",
            fontsize=9.0,
            pad=2,
        )
        ax.grid(color="0.90", linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_flat[-1].axis("off")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(legend_labels),
        frameon=False,
        fontsize=7.5,
        handlelength=1.8,
        handletextpad=0.45,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("Time (trading days)")
    fig.supylabel(ylabel, x=0.05)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(
        rect=(0.045, 0.075, 0.995, 1.0),
        h_pad=0.55,
        w_pad=0.65,
    )
    save_pdf(fig, output)

def _signature_digest(payload, arrays):
    hasher = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode())
        hasher.update(str(contiguous.shape).encode())
        hasher.update(contiguous.view(np.uint8))
    return hasher.hexdigest()


def _cached_paths(path, signature, solve, *, force):
    if not force and path.exists():
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata_json'].item()))
            if metadata['signature'] == signature:
                return {key: np.asarray(data[key]) for key in
                        ('mean_inventory', 'impact', 'sample_average_prior') if key in data}
    arrays = solve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata_json=np.array(json.dumps({'signature': signature})), **arrays)
    return arrays


def _finite_mean_field_comparison(engine, n_players, prior_pool):
    calibration, scenarios = engine.calibration, engine.scenarios
    priors = prior_pool[:n_players]
    values = {key: np.zeros((1, len(engine.grid.times), 5))
              for key in ('mean_inventory', 'impact')}
    for column, asset in enumerate(calibration.candidate_assets):
        orders = scenarios.orders[:, [asset]]
        low, high = float(orders.min()), float(orders.max())
        weights = np.column_stack((1.0 - (orders[:, 0] - low) / (high - low),
                                   (orders[:, 0] - low) / (high - low)))
        params = FiniteGameParameters(
            N=n_players,
            **{key: matrix[np.ix_([asset], [asset])] for key, matrix in
               zip(('Lambda', 'R', 'Gamma', 'Phi', 'Psi'), engine.matrices)},
            **{key: getattr(calibration, key) for key in
               ('T_ann', 'T_twap', 'T_eff', 'T', 'twap_tau', 'chi_u', 'chi_g')},
            scenario_orders=np.array([[low], [high]]), priors=priors @ weights,
            varpi_n=30.0 / n_players)
        paths = solve_common_inventory_paths(params=params, times=engine.grid.times,
                                             scenario_numbers=[1, 2],
                                             rtol=FINITE_COMPARISON_RTOL,
                                             atol=FINITE_COMPARISON_ATOL)
        for key in values:
            values[key][0, :, column] = sum(weights[1, k - 1] * paths[k][key][:, 0]
                                            for k in paths)
    values['sample_average_prior'] = priors.mean(axis=0)
    return values


def generate_mean_field_figures(engine, output_root, cache_root):
    calibration, scenarios = engine.calibration, engine.scenarios
    times, assets = engine.grid.times, calibration.candidate_assets
    event_times = {name: float(getattr(calibration, name)) for name in
                   ('T_ann', 'T_twap', 'T_eff', 'T', 'twap_tau')}
    populations = (30, 60, 150, 300)
    prior_pool = build_prior_pool(calibration, scenarios, max_players=300)
    finite = {}
    for n_players in populations:
        signature = _signature_digest(
            {**event_times, 'version': 'mean-field-figures-v7-event-times', 'n_players': n_players,
             'varpi_n': 30.0 / n_players, 'regime': 'fixed', 'scenario_numbers': (2,),
             'chi_u': calibration.chi_u, 'chi_g': calibration.chi_g, 'smoke': False,
             'rtol': FINITE_COMPARISON_RTOL, 'atol': FINITE_COMPARISON_ATOL},
            (times, scenarios.orders, scenarios.average_prior, prior_pool[:n_players],
             *engine.matrices))
        scaling_tag = 'varpi_1' if n_players == 30 else 'varpi_30_over_n'
        path = (cache_root / 'finite_common_paths' / scaling_tag /
                'fixed_penalties' / f'n{n_players:03d}_{signature[:20]}.npz')
        finite[n_players] = _cached_paths(
            path, signature, lambda: _finite_mean_field_comparison(engine, n_players, prior_pool),
            force=engine.cache.force)
    matrices = tuple(matrix[np.ix_(assets, assets)] for matrix in engine.matrices)
    scenario_numbers = (2,)
    signature = _signature_digest(
        {**event_times, 'version': 'mean-field-figures-v7-event-times', 'kind': 'mean_field_common_paths', 'varpi': 30.0,
         'scenario_numbers': scenario_numbers, 'chi_u': calibration.chi_u,
         'chi_g': calibration.chi_g, 'smoke': False},
        (times, scenarios.orders[:, assets], scenarios.average_prior, *matrices))

    def solve_mean_field():
        params = MeanFieldParameters(
            varpi=30.0,
            **dict(zip(('Lambda', 'R', 'Gamma', 'Phi', 'Psi'), matrices)),
            **{key: getattr(calibration, key) for key in
               ('T_ann', 'T_twap', 'T_eff', 'T', 'twap_tau', 'chi_u', 'chi_g')},
            scenario_orders=scenarios.orders[:, assets], average_prior=scenarios.average_prior)
        paths = solve_diagonal_mean_field_paths(params=params, times=times,
                                                scenario_numbers=list(scenario_numbers))
        return {key: np.stack([paths[k][key] for k in scenario_numbers])
                for key in ('mean_inventory', 'impact')}

    mean_field = _cached_paths(
        cache_root / 'mean_field_common_paths' / f'paths_{signature[:20]}.npz',
        signature, solve_mean_field, force=engine.cache.force)
    outputs = []
    scenario_label = scenario_title(tuple(asset + 1 for asset in scenarios.pairs[1]))
    for key, folder, prefix, title, ylabel in (
        ('mean_inventory', 'inventory', 'average_inventory', 'Average Inventory by Asset',
         'Average inventory (mn shares)'),
        ('impact', 'impact_state', 'impact_state', 'Impact by Asset', IMPACT_YLABEL)):
        series = []
        for index, n_players in enumerate(populations):
            probability = finite[n_players]['sample_average_prior'][1]
            label = rf'$n={n_players}$ ($\bar\pi_n^k={100 * probability:.2f}\%$)'
            series.append((label, finite[n_players][key][0], ASSET_COLORS[index], '-', 1.35))
        probability = scenarios.average_prior[1]
        label = rf'Mean field ($\bar\pi^k={100 * probability:.2f}\%$)'
        series.append((label, mean_field[key][0], '0.05', '--', 1.8))
        output = output_root / folder / (
            f'{prefix}_varpi_30_over_n_chi_u_1_chi_g_1_n030_060_150_300_scenario_02.pdf')
        _candidate_panel_figure(times=times, series=series, calibration=calibration,
                                candidate_assets=assets,
                                title=title + r' ($\varpi=30$)' + '\n' + scenario_label,
                                ylabel=ylabel, output=output)
        outputs.append(output)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproduce the nine figures in the current paper.')
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT.parent))
    parser.add_argument('--cache-root', default=str(DEFAULT_CACHE_ROOT.parent))
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    configure_matplotlib()
    engine = ExperimentEngine(cache_root=anchored_path(args.cache_root) / 'finite_player_game', grid=make_experiment_grid(),
                              force=args.force)
    output_root = anchored_path(args.output_root)
    outputs = generate_manuscript_figures(engine, output_root / 'finite_player_game')
    outputs.extend(generate_mean_field_figures(
        engine, output_root / 'mean_field_game',
        anchored_path(args.cache_root) / 'mean_field_game'))
    print(f'Generated {len(outputs)} manuscript figures.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
