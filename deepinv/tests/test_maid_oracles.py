"""Rung 2: HypergradientOracle seam, certification rule, certification rule.

Smooth bound probe (144 configurations) must never under-estimate, matching
the supervisor's acceptance criterion for rung 1 after the refactor.
"""

from __future__ import annotations

import pytest
import torch

from deepinv.optim.bilevel import (
    MAID,
    MAIDConfig,
    SmoothHypergradientOracle,
    hypergradient_error_bound,
    strong_convexity_distance_bound,
)
from deepinv.optim.bilevel.oracle import HypergradientOracle
from deepinv.tests.test_maid import _make_section41_problem

# ---------------------------------------------------------------------------
# Certification rule
# ---------------------------------------------------------------------------


class _UncertifiedOracle(HypergradientOracle):
    """Minimal non-certified stub used only to test the opt-in gate."""

    @property
    def certified(self) -> bool:
        return False

    @property
    def citation(self) -> str:
        return ""

    @property
    def L_g(self) -> float:
        return 1.0

    def solve_lower_level(self, theta, eps, warm_start=None):
        raise NotImplementedError

    def hypergradient(self, theta, lower, delta):
        raise NotImplementedError

    def error_bound(self, theta, lower, hyper, eps, delta):
        return 0.0

    def g(self, x):
        return torch.tensor(0.0)

    def grad_g(self, x):
        return torch.zeros_like(x)


def test_uncertified_oracle_rejected_by_default():
    with pytest.raises(ValueError, match="not certified"):
        MAID(_UncertifiedOracle())


def test_uncertified_oracle_accepted_with_opt_in():
    maid = MAID(_UncertifiedOracle(), allow_uncertified=True)
    assert maid.oracle.certified is False


def test_smooth_oracle_is_certified():
    problem, _, _ = _make_section41_problem()
    oracle = SmoothHypergradientOracle(problem)
    assert oracle.certified is True
    assert "2308.10098" in oracle.citation
    MAID(oracle)  # must not raise


def test_strong_convexity_distance_bound_identity():
    """Lemma 1: ||x* - x|| <= ||grad Phi(x)|| / mu for a quadratic Phi."""
    mu = 2.5
    x_star = torch.tensor([1.0, -2.0], dtype=torch.float64)
    x = torch.tensor([0.3, 0.4], dtype=torch.float64)
    # Phi(x) = (mu/2)||x - x_star||^2, grad = mu (x - x_star)
    grad = mu * (x - x_star)
    bound = strong_convexity_distance_bound(float(grad.norm().item()), mu)
    true_dist = float((x - x_star).norm().item())
    assert bound >= true_dist - 1e-12
    assert abs(bound - true_dist) < 1e-12  # equality for quadratics


# ---------------------------------------------------------------------------
# Smooth 144-configuration probe (supervisor acceptance criterion)
# ---------------------------------------------------------------------------


def test_smooth_error_bound_never_underestimates_144():
    """omega never under-estimates true error across 144 configurations.

    Four seeds, three parameter scales, four eps, three delta. This is the
    probe the supervisor ran on rung 1; it must remain green after the
    oracle refactor.
    """
    seeds = [0, 1, 2, 3]
    scales = [0.5, 1.0, 2.0]
    eps_list = [1e-1, 1e-2, 1e-3, 1e-4]
    delta_list = [1e-1, 1e-2, 1e-3]

    violations = 0
    ratios = []
    n_configs = 0

    for seed in seeds:
        for scale in scales:
            problem, theta0, _ = _make_section41_problem(seed=seed + 10)
            # Scale the starting point so the geometry varies.
            theta = scale * theta0
            z_exact = problem.exact_hypergradient(theta, problem.closed_form_x(theta))
            for eps in eps_list:
                for delta in delta_list:
                    n_configs += 1
                    xbar, _ = problem.solve_lower(theta, eps=eps)
                    z, _ = problem.inexact_hypergradient(xbar, theta, delta=delta)
                    true_err = float((z - z_exact).norm().item())
                    omega = hypergradient_error_bound(
                        eps=eps,
                        delta=delta,
                        mu=problem.mu,
                        L_g=problem.L_g,
                        J_norm=problem.J_norm,
                        grad_g_norm=float(problem.grad_g(xbar).norm().item()),
                        L_H_inv=0.0,
                        L_J=0.0,
                    )
                    if omega < true_err - 1e-12:
                        violations += 1
                    if true_err > 1e-15:
                        ratios.append(omega / true_err)

    assert n_configs == 144
    assert violations == 0, f"{violations} of 144 bound violations"
    assert min(ratios) >= 1.0 - 1e-9
    # Loose is acceptable; under-estimate is not.
    assert max(ratios) > 1.0


def test_smooth_oracle_error_bound_matches_function():
    """Oracle.error_bound equals the free function on the same inputs."""
    problem, theta0, _ = _make_section41_problem()
    oracle = SmoothHypergradientOracle(problem)
    eps, delta = 1e-2, 1e-2
    lower = oracle.solve_lower_level(theta0, eps=eps)
    hyper = oracle.hypergradient(theta0, lower, delta=delta)
    omega_oracle = oracle.error_bound(theta0, lower, hyper, eps, delta)
    J_norm = problem.estimate_J_norm(lower.x, theta0)
    omega_fn = hypergradient_error_bound(
        eps=eps,
        delta=delta,
        mu=problem.mu,
        L_g=problem.L_g,
        J_norm=J_norm,
        grad_g_norm=float(problem.grad_g(lower.x).norm().item()),
        L_H_inv=0.0,
        L_J=0.0,
    )
    assert abs(omega_oracle - omega_fn) < 1e-15


def test_maid_default_skips_descent_test_and_instruments_failures():
    """Default config has check_descent_direction=False and exposes counters."""
    problem, theta0, theta_star = _make_section41_problem()
    config = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=1.0
        / float(
            2.0
            * torch.linalg.eigvalsh(
                (problem.A1 @ problem._P @ problem.A3).T
                @ (problem.A1 @ problem._P @ problem.A3)
            )[-1].item()
        ),
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=30,
        max_iter=40,
        tol=1e-6,
        g_convex=True,
        check_descent_direction=False,
    )
    assert config.check_descent_direction is False
    maid = MAID(problem, config)
    theta_final, history = maid.run(theta0)
    assert all(o != o for o in history["omega"])  # all nan
    assert "n_backtrack_failures" in history
    assert "n_lower_solves" in history
    assert history["n_lower_solves"] > 0
    assert history["n_hypergradients"] > 0
    assert history["n_upper_iters"] == len(history["f_exact"])
    f_final = float(problem.f_closed_form(theta_final).item())
    f_star = float(problem.f_closed_form(theta_star).item())
    assert (f_final - f_star) / max(abs(f_star), 1.0) < 1e-3


def test_maid_certified_descent_path_still_works():
    """check_descent_direction=True restores Algorithm 3.2 refinement."""
    problem, theta0, theta_star = _make_section41_problem()
    Lf = float(
        2.0
        * torch.linalg.eigvalsh(
            (problem.A1 @ problem._P @ problem.A3).T
            @ (problem.A1 @ problem._P @ problem.A3)
        )[-1].item()
    )
    config = MAIDConfig(
        eps0=1e-1,
        delta0=1e-1,
        alpha0=1.0 / Lf,
        rho=0.5,
        rho_bar=1.5,
        nu=0.5,
        nu_bar=1.1,
        eta=0.5,
        lambd=0.1,
        max_BT=30,
        max_iter=40,
        tol=1e-6,
        g_convex=True,
        check_descent_direction=True,
    )
    maid = MAID(problem, config)
    theta_final, history = maid.run(theta0)
    assert all(o == o for o in history["omega"])  # finite
    f_final = float(problem.f_closed_form(theta_final).item())
    f_star = float(problem.f_closed_form(theta_star).item())
    assert (f_final - f_star) / max(abs(f_star), 1.0) < 1e-3
