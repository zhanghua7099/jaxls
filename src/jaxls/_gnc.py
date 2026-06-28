"""Graduated Non-Convexity (GNC) solver for robust estimation.

Implements the GNC algorithm from Yang et al. (2020), "Graduated
Non-Convexity for Robust Spatial Perception" (IEEE RA-L).  GNC wraps
the standard jaxls least-squares solver in an outer loop that gradually
increases non-convexity, driving outlier weights toward zero while
keeping inlier weights near one.

Two kernels are supported:

- **TLS** (Truncated Least Squares): hard inlier/outlier classification.
  Weights converge to binary {0, 1}.
- **GM** (Geman-McClure): soft redescending influence.

Weights are computed **per measurement** (on ``‖r_i‖²``, the squared
norm of each measurement's residual vector), consistent with the
original GNC formulation.  For scalar residuals (``residual_flat_dim=1``)
this is identical to element-wise weighting.

Usage::

    solution, summary = jaxls.gnc_solve(
        costs, variables,
        gnc_config=jaxls.GNCConfig(noise_bound=0.5, kernel="tls"),
    )
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Literal

import jax
import jax.numpy as jnp
import numpy as onp
from loguru import logger

from ._cost import Cost
from ._problem import AnalyzedLeastSquaresProblem, LeastSquaresProblem
from ._solvers import (
    ConjugateGradientConfig,
    SolveSummary,
    TerminationConfig,
    TrustRegionConfig,
)
from ._variables import Var, VarValues


@dataclasses.dataclass
class GNCSummary:
    """Summary of a GNC solve."""

    weights: jax.Array
    """Final per-measurement weights (one scalar per measurement)."""
    mu_history: list[float]
    """Mu value at each GNC outer iteration."""
    cost_history: list[float]
    """Unweighted cost at each GNC outer iteration."""
    gnc_iterations: int
    """Number of GNC outer iterations executed."""
    inner_summaries: list[SolveSummary]
    """Inner solver summary for each GNC iteration."""


@dataclasses.dataclass
class GNCConfig:
    """Configuration for the GNC outer loop.

    Args:
        kernel: Robust kernel type.  ``"tls"`` for Truncated Least Squares
            (binary inlier/outlier), ``"gm"`` for Geman-McClure (soft).
        noise_bound: Inlier noise bound in residual units.  For a
            measurement with residual vector ``r``, it is classified as an
            inlier when ``‖r‖ <= noise_bound``.  Internally squared to
            ``barc2 = noise_bound ** 2`` and compared against ``‖r‖²``.

            This is a *confidence threshold*, not the noise standard
            deviation itself.  A typical choice is
            ``noise_bound = k * noise_std`` where ``k`` controls the
            confidence level (k=2 for ~95%, k=3 for ~99.7%).  Setting
            ``noise_bound = noise_std`` (k=1) is too aggressive: ~32% of
            genuine inliers would exceed the bound due to normal noise
            fluctuations and be incorrectly rejected.

            For multi-dimensional residuals the comparison is on the
            squared *norm* ``‖r‖²``, so ``noise_bound`` should be set
            according to the chi-squared distribution with
            ``residual_dim`` degrees of freedom.
        continuation_factor: Multiplicative factor for mu progression.
            Default ``1.4`` (from Yang et al.).
        max_iterations: Maximum number of GNC outer iterations.
        weight_tolerance: Tolerance for binary weight convergence check.
            Weights within this tolerance of 0 or 1 are considered binary.
        cost_tolerance: Relative cost change threshold for early stopping.
        initial_mu: Starting value of the shape parameter.  ``None``
            (default) auto-computes from the initial residuals.
    """

    kernel: Literal["tls", "gm"] = "tls"
    noise_bound: float = 1.0
    continuation_factor: float = 1.4
    max_iterations: int = 100
    weight_tolerance: float = 1e-4
    cost_tolerance: float = 1e-6
    initial_mu: float | None = None


def _gnc_tls_weights(
    residuals_sq: jax.Array, mu: float, barc2: float
) -> jax.Array:
    """GNC-TLS weight from per-measurement ‖r‖²."""
    th1 = (mu + 1.0) / mu * barc2
    th2 = mu / (mu + 1.0) * barc2
    w_mid = jnp.sqrt(barc2 * mu * (mu + 1.0) / jnp.maximum(residuals_sq, 1e-20)) - mu
    w_mid = jnp.clip(w_mid, 0.0, 1.0)
    return jnp.where(residuals_sq <= th2, 1.0, jnp.where(residuals_sq >= th1, 0.0, w_mid))


def _gnc_gm_weights(
    residuals_sq: jax.Array, mu: float, barc2: float
) -> jax.Array:
    """GNC-GM weight from per-measurement ‖r‖²."""
    return (mu * barc2 / (residuals_sq + mu * barc2)) ** 2


def _auto_initial_mu_tls(max_res_sq: float, barc2: float) -> float:
    denom = 2.0 * max_res_sq / barc2 - 1.0
    return max(1.0 / max(denom, 1e-10), 1e-6)


def _auto_initial_mu_gm(max_res_sq: float, barc2: float) -> float:
    return max(2.0 * max_res_sq / barc2, 1e-6)


def _weights_are_binary(weights: jax.Array, tol: float) -> bool:
    w = onp.asarray(weights)
    return bool(onp.all((w < tol) | (w > 1.0 - tol)))


def _make_gnc_weight_fn(
    kernel: Literal["tls", "gm"], mu: float, barc2: float
) -> Any:
    """Build an ``irls_weight_fn`` that computes per-measurement GNC weights.

    The returned function receives residuals with shape
    ``(count, residual_flat_dim)`` and returns weights of the same shape.
    Internally it computes ``‖r_i‖² = sum(r_i²)`` per measurement (row),
    derives a single scalar weight from the GNC formula, and broadcasts
    that weight to every element of the measurement.
    """
    kernel_fn = _gnc_tls_weights if kernel == "tls" else _gnc_gm_weights

    def weight_fn(residuals: jax.Array) -> jax.Array:
        # Per-measurement squared norm: (count, dim) -> (count, 1)
        r_sq = jnp.sum(residuals**2, axis=-1, keepdims=True)
        w = kernel_fn(r_sq, mu, barc2)  # (count, 1)
        return jnp.broadcast_to(w, residuals.shape)

    return weight_fn


def _apply_gnc_weights(
    costs: list[Cost],
    robust_mask: list[bool],
    weight_fn: Any,
) -> list[Cost]:
    """Return a new cost list with GNC weight function applied to robust costs."""
    new_costs: list[Cost] = []
    for cost, is_robust in zip(costs, robust_mask):
        if is_robust:
            new_costs.append(
                Cost(
                    compute_residual=cost.compute_residual,
                    args=cost.args,
                    kind=cost.kind,
                    jac_mode=cost.jac_mode,
                    jac_batch_size=cost.jac_batch_size,
                    jac_custom_fn=cost.jac_custom_fn,
                    jac_custom_with_cache_fn=cost.jac_custom_with_cache_fn,
                    irls_weight_fn=weight_fn,
                    name=cost.name,
                )
            )
        else:
            new_costs.append(cost)
    return new_costs


def _compute_per_measurement_res_sq(
    problem: AnalyzedLeastSquaresProblem,
    vals: VarValues,
) -> jax.Array:
    """Compute per-measurement ‖r_i‖² from an analyzed problem.

    Returns a 1-D array with one entry per measurement (cost instance),
    where each entry is the squared norm of that measurement's residual
    vector.
    """
    residual_vector = problem.compute_residual_vector(vals)
    pieces: list[jax.Array] = []
    offset = 0
    for stacked_cost, count in zip(problem._stacked_costs, problem._cost_counts):
        dim = stacked_cost.residual_flat_dim
        block = residual_vector[offset : offset + count * dim]
        # (count, dim) -> sum over dim -> (count,)
        pieces.append(jnp.sum(block.reshape(count, dim) ** 2, axis=-1))
        offset += count * dim
    return jnp.concatenate(pieces, axis=0)


def gnc_solve(
    costs: Iterable[Cost],
    variables: Iterable[Var],
    initial_vals: VarValues | None = None,
    gnc_config: GNCConfig | None = None,
    *,
    linear_solver: (
        Literal["conjugate_gradient", "cholmod", "dense_cholesky"]
        | ConjugateGradientConfig
    ) = "conjugate_gradient",
    trust_region: TrustRegionConfig | None = TrustRegionConfig(),
    termination: TerminationConfig = TerminationConfig(),
    verbose: bool = True,
    return_summary: bool = True,
    **solve_kwargs: Any,
) -> tuple[VarValues, GNCSummary] | VarValues:
    """Solve a least-squares problem with GNC robust outlier rejection.

    GNC wraps the standard jaxls solver in an outer loop that gradually
    increases non-convexity.  At each outer iteration the solver runs to
    convergence with IRLS weights derived from the current mu value,
    then mu is updated and the weights are recomputed.

    GNC weights are applied to every ``l2_squared`` cost that does **not**
    already have an ``irls_weight_fn`` set.  Costs with an existing weight
    function (e.g. priors) and constraint costs are left unchanged.

    Weights are computed **per measurement** on the squared residual norm
    ``‖r_i‖²``, consistent with the original GNC formulation (Yang et al.
    2020).  For scalar residuals this is equivalent to element-wise
    weighting.

    Args:
        costs: Cost terms (same as ``LeastSquaresProblem``).
        variables: Variables to optimise.
        initial_vals: Starting point.  If ``None``, default values are used.
        gnc_config: GNC parameters.  If ``None``, uses ``GNCConfig()``.
        linear_solver: Passed through to the inner solver.
        trust_region: Passed through to the inner solver.
        termination: Passed through to the inner solver.
        verbose: Print GNC progress.
        return_summary: If ``True`` (default), return ``(vals, summary)``.
        **solve_kwargs: Extra keyword arguments forwarded to
            ``AnalyzedLeastSquaresProblem.solve``.

    Returns:
        Optimised variable values, and optionally a :class:`GNCSummary`.
    """
    if gnc_config is None:
        gnc_config = GNCConfig()

    costs_list = list(costs)
    variables_list = list(variables)
    barc2 = gnc_config.noise_bound ** 2
    kernel = gnc_config.kernel

    robust_mask = [
        c.kind == "l2_squared" and c.irls_weight_fn is None for c in costs_list
    ]
    n_robust = sum(robust_mask)
    if n_robust == 0:
        logger.warning(
            "GNC: no robust costs found (all costs either have irls_weight_fn "
            "or are constraints). Running standard solve."
        )
        problem = LeastSquaresProblem(costs_list, variables_list).analyze()
        result = problem.solve(
            initial_vals,
            linear_solver=linear_solver,
            trust_region=trust_region,
            termination=termination,
            verbose=verbose,
            return_summary=True,
            **solve_kwargs,
        )
        assert isinstance(result, tuple)
        vals, inner_summary = result
        summary = GNCSummary(
            weights=jnp.ones(0),
            mu_history=[],
            cost_history=[],
            gnc_iterations=0,
            inner_summaries=[inner_summary],
        )
        return (vals, summary) if return_summary else vals

    # --- Step 0: initial solve (all weights = 1) ---
    if verbose:
        logger.info(
            "GNC ({}, noise_bound={:.4f}): {} robust costs, {} total costs",
            kernel.upper(),
            gnc_config.noise_bound,
            n_robust,
            len(costs_list),
        )

    problem = LeastSquaresProblem(costs_list, variables_list).analyze()
    result = problem.solve(
        initial_vals,
        linear_solver=linear_solver,
        trust_region=trust_region,
        termination=termination,
        verbose=verbose,
        return_summary=True,
        **solve_kwargs,
    )
    assert isinstance(result, tuple)
    vals, inner_summary = result
    inner_summaries: list[SolveSummary] = [inner_summary]

    # Compute initial per-measurement ‖r_i‖² for auto mu.
    per_meas_res_sq = _compute_per_measurement_res_sq(problem, vals)
    max_res_sq = float(jnp.max(per_meas_res_sq))

    # Auto-compute initial mu.
    if gnc_config.initial_mu is not None:
        mu = gnc_config.initial_mu
    elif kernel == "tls":
        mu = _auto_initial_mu_tls(max_res_sq, barc2)
    else:
        mu = _auto_initial_mu_gm(max_res_sq, barc2)

    if verbose:
        logger.info("GNC: initial mu = {:.6f}", mu)

    mu_history = [mu]
    prev_unweighted_cost = float(jnp.sum(per_meas_res_sq))
    cost_history = [prev_unweighted_cost]

    # --- GNC outer loop ---
    kernel_fn = _gnc_tls_weights if kernel == "tls" else _gnc_gm_weights
    weights = jnp.ones_like(per_meas_res_sq)
    for gnc_iter in range(gnc_config.max_iterations):
        weight_fn = _make_gnc_weight_fn(kernel, mu, barc2)
        gnc_costs = _apply_gnc_weights(costs_list, robust_mask, weight_fn)

        problem = LeastSquaresProblem(gnc_costs, variables_list).analyze()
        result = problem.solve(
            vals,
            linear_solver=linear_solver,
            trust_region=trust_region,
            termination=termination,
            verbose=False,
            return_summary=True,
            **solve_kwargs,
        )
        assert isinstance(result, tuple)
        vals, inner_summary = result
        inner_summaries.append(inner_summary)

        # Per-measurement ‖r_i‖² for convergence checks.
        unweighted_problem = LeastSquaresProblem(costs_list, variables_list).analyze()
        per_meas_res_sq = _compute_per_measurement_res_sq(unweighted_problem, vals)
        weights = kernel_fn(per_meas_res_sq, mu, barc2)
        unweighted_cost = float(jnp.sum(per_meas_res_sq))
        cost_history.append(unweighted_cost)

        if verbose:
            n_inliers = int(jnp.sum(weights > 0.5))
            n_total = len(weights)
            logger.info(
                "GNC iter {}: mu={:.6f}  cost={:.6f}  inliers={}/{}",
                gnc_iter + 1,
                mu,
                unweighted_cost,
                n_inliers,
                n_total,
            )

        # Convergence checks.
        binary = _weights_are_binary(weights, gnc_config.weight_tolerance)
        if binary:
            if verbose:
                logger.info("GNC converged: weights are binary at iter {}", gnc_iter + 1)
            break

        rel_change = abs(cost_history[-1] - cost_history[-2]) / max(
            abs(cost_history[-2]), 1e-10
        )
        if rel_change < gnc_config.cost_tolerance:
            if verbose:
                logger.info(
                    "GNC converged: cost change {:.2e} < tol at iter {}",
                    rel_change,
                    gnc_iter + 1,
                )
            break

        # For GM, also stop when mu has decreased to 1.0 (the true GM kernel).
        if kernel == "gm" and mu <= 1.0 + 1e-6:
            if verbose:
                logger.info("GNC-GM converged: mu reached 1.0 at iter {}", gnc_iter + 1)
            break

        # Update mu.
        if kernel == "tls":
            mu = mu * gnc_config.continuation_factor
        else:
            mu = mu / gnc_config.continuation_factor
        mu_history.append(mu)

    summary = GNCSummary(
        weights=weights,
        mu_history=mu_history,
        cost_history=cost_history,
        gnc_iterations=gnc_iter + 1 if gnc_config.max_iterations > 0 else 0,
        inner_summaries=inner_summaries,
    )
    if return_summary:
        return vals, summary
    return vals
