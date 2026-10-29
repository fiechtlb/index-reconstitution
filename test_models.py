"""Checks against manuscript coefficients, boundary conditions, and independent dynamics."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm

from _index_rebalance_common import FiniteGameParameters, MeanFieldParameters, make_time_grid
from experiments import (
    EquilibriumSummary, baseline_calibration, build_prior_pool, build_scenarios,
    calibrated_matrices, impact_state_event_metrics, player_netted_timing_metrics,
    scenario_indexer_cost,
)
from index_rebalance_finite_closed_loop import (
    finite_matrices, simulate_finite_scenarios, solve_common_inventory_paths, solve_finite_game,
)
from index_rebalance_mean_field import solve_diagonal_mean_field_paths


def finite_parameters(n=3, *, varpi_n=1.0):
    return FiniteGameParameters(
        N=n, Lambda=np.diag([0.8]), R=np.diag([0.7]), Gamma=np.diag([0.3]),
        Phi=np.diag([0.2]), Psi=np.diag([3.0]), T_ann=1.0, T_twap=1.5,
        T_eff=2.0, T=2.5, twap_tau=0.5, scenario_orders=np.array([[1.0], [-0.6]]),
        priors=np.array([[0.7 - 0.1*(i % 2), 0.3 + 0.1*(i % 2)] for i in range(n)]),
        varpi_n=varpi_n, chi_u=0.7, chi_g=0.6)


def model_times(points=31):
    return make_time_grid(2.5, 1.0, 1.5, 2.0, n_pre=points,
                          n_announcement_to_twap=points, n_twap=points, n_post=points)


def exact_mean_field_paths(params, times):
    """Solve the scalar first-order conditions by exact phase transitions.

    Coordinates are inventory, impact, and inventory adjoint. Unknown adjoints
    are fixed by terminal inventory penalties and subjective announcement matching.
    This reference uses dense matrix exponentials rather than time discretization.
    """
    lam, rho, gamma, phi, psi = (float(getattr(params, name)[0, 0])
                                for name in ("Lambda", "R", "Gamma", "Phi", "Psi"))
    inverse = 1.0 / ((2 + params.varpi*params.chi_u)*lam)
    drift = np.array([[0, -inverse, -inverse],
                      [0, -rho - gamma*params.varpi*inverse, -gamma*params.varpi*inverse],
                      [-phi, 0, 0]])
    loading = np.array([-params.chi_g*lam*inverse,
                        gamma*(1 - params.varpi*params.chi_g*lam*inverse), 0])

    def advance(initial, start, stop, scenario):
        state = np.r_[initial, 1.0]
        boundaries = sorted({start, stop, *(event for event in (params.T_twap, params.T_eff)
                                            if start < event < stop)})
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            augmented = np.zeros((4, 4))
            augmented[:3, :3] = drift
            g = params.g(scenario, (left + right)/2)[0]
            augmented[:3, 3] = loading*g
            state = expm((right-left)*augmented) @ state
        return state[:3]

    def residual(adjoint):
        announcement = advance([0, 0, adjoint[0]], 0, params.T_ann, 0)
        result = [announcement[2] - params.average_prior @ adjoint[1:]]
        for k in range(1, params.K+1):
            terminal = advance([*announcement[:2], adjoint[k]], params.T_ann, params.T, k)
            result.append(terminal[2] - psi*terminal[0])
        return np.array(result)

    offset = residual(np.zeros(params.K+1))
    matrix = np.column_stack([residual(basis)-offset for basis in np.eye(params.K+1)])
    adjoint = np.linalg.solve(matrix, -offset)
    announcement = advance([0, 0, adjoint[0]], 0, params.T_ann, 0)
    return {k: np.array([advance([0, 0, adjoint[0]], 0, t, 0) if t < params.T_ann
                         else advance([*announcement[:2], adjoint[k]], params.T_ann, t, k)
                         for t in times]) for k in range(1, params.K+1)}


class ManuscriptModelTest(unittest.TestCase):
    def test_calibration_and_reference_price_rebalance(self):
        calibration = baseline_calibration()
        scenarios = build_scenarios(calibration)
        lam, rho, gamma, q, q_terminal = calibrated_matrices(calibration)
        np.testing.assert_allclose(np.diag(lam), np.array([0.44, 0.244, 2.5, 3.01, 10.6])/6.5)
        np.testing.assert_allclose(np.diag(gamma), [1.09, 0.63, 4.5, 6.9, 25.3])
        np.testing.assert_allclose(np.diag(rho), 2*np.log(2))
        np.testing.assert_allclose(q, 0.25*lam)
        np.testing.assert_allclose(q_terminal, 13000*lam)
        np.testing.assert_array_equal(calibration.float_shares, 100*calibration.adv)
        np.testing.assert_allclose(scenarios.initial_holdings, 7*calibration.adv*[1, 1, 0, 0, 0])
        self.assertAlmostEqual(calibration.prices @ scenarios.initial_holdings, 2111.62702436, places=9)
        self.assertEqual(scenarios.pairs[1], (0, 2))
        self.assertEqual(len(scenarios.pairs), 10)
        weights = np.array([(5-a)*(5-b) for a, b in scenarios.pairs], dtype=float)
        np.testing.assert_allclose(scenarios.average_prior, weights/weights.sum())
        np.testing.assert_allclose(scenarios.orders @ calibration.prices, 0, atol=1e-11)
        short = build_prior_pool(calibration, scenarios, max_players=5)
        np.testing.assert_array_equal(short, build_prior_pool(calibration, scenarios, max_players=20)[:5])

    def test_population_weight_domain(self):
        params = finite_parameters(varpi_n=0.5)
        self.assertAlmostEqual(params.vartheta_n, 0.35)
        with self.assertRaisesRegex(ValueError, "chi_u.*varpi_n"):
            replace(params, chi_u=2.01)
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "varpi_n"):
                replace(params, varpi_n=value)

    def test_batched_finite_paths_match_common_inventory_solution(self):
        times = model_times()
        for weight in (1.0, 1/3, 2/3):
            with self.subTest(varpi_n=weight):
                params = finite_parameters(varpi_n=weight)
                paths = simulate_finite_scenarios(solve_finite_game(params, times, solve_alpha=False), times)
                common = solve_common_inventory_paths(params=params, times=times, scenario_numbers=[1, 2])
                for k, path in enumerate(paths, 1):
                    for field in ("mean_inventory", "impact"):
                        np.testing.assert_allclose(path[field], common[k][field], atol=3e-6, rtol=3e-5)

    def test_announcement_and_terminal_value_matching(self):
        params = finite_parameters()
        solution = solve_finite_game(params, model_times())
        np.testing.assert_allclose(solution.P[-1], [[3, 3, 0], [3, 3, 0], [0, 0, 0]], atol=1e-13)
        np.testing.assert_allclose(solution.ell_post[:, -1], 0, atol=1e-13)
        np.testing.assert_allclose(solution.alpha_post[:, -1], 0, atol=1e-13)
        np.testing.assert_allclose(solution.ell_pre[-1], params.priors @ solution.ell_post[:, 0], atol=1e-13)
        np.testing.assert_allclose(solution.alpha_pre[-1], params.priors @ solution.alpha_post[:, 0], atol=1e-13)

    def test_dense_multi_asset_hjb_at_arbitrary_states(self):
        params = FiniteGameParameters(
            N=3, Lambda=np.array([[.8, .12], [.12, .6]]), R=.7*np.eye(2),
            Gamma=np.array([[.3, .07], [.07, .25]]), Phi=np.array([[.2, -.03], [-.03, .15]]),
            Psi=np.array([[3., .4], [.4, 2.]]), T_ann=1., T_twap=1.5, T_eff=2., T=2.5,
            twap_tau=.5, scenario_orders=np.array([[1., -.4], [-.6, .7]]),
            priors=np.array([[.7, .3], [.6, .4], [.4, .6]]), varpi_n=.4, chi_u=.7, chi_g=.6)
        times = make_time_grid(2.5, 1., 1.5, 2., n_pre=1001,
                               n_announcement_to_twap=501, n_twap=501, n_post=501)
        solution = solve_finite_game(params, times, rtol=1e-10, atol=1e-12)
        matrices = finite_matrices(params)
        inventory = np.array([[.2, -.5], [-.3, .1], [.4, .2]])
        impact = np.array([.17, -.23])
        average = inventory.mean(axis=0)
        states = np.c_[inventory-average, np.tile(average, (3, 1)), np.tile(impact, (3, 1))]
        for time, scenario in ((.4, 0), (1.25, 1), (1.75, 2), (2.25, 1)):
            with self.subTest(time=time, scenario=scenario):
                index = np.flatnonzero(np.isclose(times, time))[0]
                P = solution.P_at(time)
                P_dot = (solution.P[index+1]-solution.P[index-1])/(times[index+1]-times[index-1])
                if scenario == 0:
                    segment, ell_values, constants = solution.pre_times, solution.ell_pre, solution.alpha_pre
                else:
                    segment = solution.post_times
                    ell_values = np.repeat(solution.ell_post[scenario-1, :, None], 3, axis=1)
                    constants = np.repeat(solution.alpha_post[scenario-1, :, None], 3, axis=1)
                j = np.flatnonzero(np.isclose(segment, time))[0]
                ell = ell_values[j]
                span = segment[j+1]-segment[j-1]
                ell_dot, constant_dot = ((values[j+1]-values[j-1])/span for values in (ell_values, constants))
                g = params.g(scenario, time)
                eta = ell@matrices.B_u + params.chi_g*(params.Lambda@g)[None]
                xi = eta/matrices.a - params.vartheta_n*eta.sum(axis=0)[None]/(matrices.a*matrices.d)
                theta = matrices.E_I + matrices.B_u.T@P
                controls = -np.linalg.solve(params.Lambda, (states@(theta@matrices.D).T+xi).T).T
                aggregate = controls.sum(axis=0)
                for player, u in enumerate(controls):
                    opponents = aggregate-u
                    gradient = P@states[player]+ell[player]
                    # Assemble drift and own-control derivative from physical state dynamics.
                    drift = np.r_[u-aggregate/params.N, aggregate/params.N,
                                  -params.R@impact+params.Gamma@(params.varpi_n*aggregate+g)]
                    own_gradient = ((1-1/params.N)*gradient[:2]+gradient[2:4]/params.N
                                    +params.varpi_n*params.Gamma.T@gradient[4:])
                    foc = (2*params.Lambda@u+params.vartheta_n*params.Lambda@opponents
                           +params.chi_g*params.Lambda@g+impact+own_gradient)
                    running = (impact+params.Lambda@(u+params.vartheta_n*opponents+params.chi_g*g))@u
                    running += .5*inventory[player]@params.Phi@inventory[player]
                    residual = (.5*states[player]@P_dot@states[player]+ell_dot[player]@states[player]
                                +constant_dot[player]+running+gradient@drift)
                    self.assertLess(abs(residual), 2e-6)
                    np.testing.assert_allclose(foc, 0, atol=1e-12, rtol=0)

    def test_belief_dispersion_preserves_aggregate_paths(self):
        times, params = model_times(), finite_parameters()
        homogeneous = replace(params, priors=np.tile(params.priors.mean(axis=0), (params.N, 1)))
        paths = [simulate_finite_scenarios(solve_finite_game(p, times, solve_alpha=False), times)
                 for p in (params, homogeneous)]
        for original, averaged in zip(*paths):
            for field in ("mean_inventory", "impact"):
                np.testing.assert_allclose(original[field], averaged[field], atol=4e-7, rtol=2e-5)
        self.assertGreater(np.max(np.abs(paths[0][0]["X"] - paths[1][0]["X"])), 1e-3)

    def test_mean_field_against_exact_phase_solution(self):
        finite = finite_parameters()
        params = MeanFieldParameters(**{name: getattr(finite, name) for name in (
            "Lambda", "R", "Gamma", "Phi", "Psi", "T_ann", "T_twap", "T_eff", "T",
            "twap_tau", "scenario_orders")}, average_prior=finite.priors.mean(axis=0),
            varpi=4, chi_u=1.3, chi_g=1.6)
        times = model_times(13)
        expected = exact_mean_field_paths(params, times)
        paths = solve_diagonal_mean_field_paths(params=params, times=times, scenario_numbers=[1, 2])
        for k, path in paths.items():
            np.testing.assert_allclose(path["mean_inventory"], expected[k][:, :1], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(path["impact"], expected[k][:, 1:2], atol=1e-10, rtol=1e-10)
        errors = []
        for n in (20, 200):
            atomic = replace(finite_parameters(n, varpi_n=params.varpi/n),
                             priors=np.tile(params.average_prior, (n, 1)), chi_u=params.chi_u, chi_g=params.chi_g)
            path = solve_common_inventory_paths(params=atomic, times=model_times(101), scenario_numbers=[1])[1]
            reference = exact_mean_field_paths(params, path["times"])[1]
            errors.append(max(np.max(np.abs(path["mean_inventory"] - reference[:, :1])),
                              np.max(np.abs(path["impact"] - reference[:, 1:2]))))
        self.assertLess(errors[1], errors[0]/5)
        self.assertLess(errors[1], 0.003)


class FigureMetricsTest(unittest.TestCase):
    def test_indexer_cost_uses_actual_population_weight(self):
        params = finite_parameters(4, varpi_n=0.5)
        edges = model_times()
        times = np.unique(np.r_[edges, (edges[:-1]+edges[1:])/2])
        flow = np.array([params.g(1, t) for t in times])
        path = dict(times=times, controls=np.full((len(times), 4, 1), 0.75),
                    impact=np.full((len(times), 1), 0.4), indexer_flow=flow)
        expected = 0.5*2*(0.4 + 0.8*(0.5*3 + 2))
        self.assertAlmostEqual(scenario_indexer_cost(params=params, path=path, edges=edges), expected)

    def test_inventory_differences_and_impact_norms_precede_averaging(self):
        inventory = np.array([[[0.5, -1], [2, 1], [-0.5, 0], [0, 0]],
                              [[-1, 0.25], [-1.5, 2], [-0.25, 0.5], [0, 0]]])
        summary = EquilibriumSummary(n_players=2, times=np.arange(4.), mean_inventory=inventory,
                                     impact=inventory, indexer_total_cost=np.zeros(2),
                                     sample_average_prior=np.array([0.25, 0.75]))
        kwargs = dict(summary=summary, average_prior=summary.sample_average_prior, candidate_assets=(0, 1))
        metrics = player_netted_timing_metrics(**kwargs, T_ann=0., T_twap=1., T_eff=2.)
        np.testing.assert_allclose([metrics[k] for k in ("inventory_at_announcement", "inventory_at_twap",
                                                       "inventory_change_during_twap")], [2.625, 6.75, 5.875])
        impact = impact_state_event_metrics(**kwargs, event_times={"announcement": 0., "twap": 1.})
        self.assertEqual(impact, {"announcement": 1.3125, "twap": 3.375})
        with self.assertRaisesRegex(RuntimeError, "event time 0.5"):
            impact_state_event_metrics(**kwargs, event_times={"missing": 0.5})


if __name__ == "__main__":
    unittest.main()
