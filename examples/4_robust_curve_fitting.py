"""Robust exponential fitting: fixed-delta vs adaptive IRLS.

Fits y = a * exp(b * x) + c to data contaminated with outliers, comparing:
  1. Standard least squares (no robustness)
  2. Fixed-delta IRLS  (requires c = c_tuning * sigma)
  3. Adaptive IRLS      (threshold-free)

Since r = y - f(x, p), we have dr/dy = 1, so sigma_r = sigma_y exactly.
This makes it the cleanest test of the fixed-scale theory.

Usage:
    python examples/4_robust_ellipse_fitting.py
"""

from loguru import logger

logger.disable("jaxls")

import jax
import jax.numpy as jnp
import jaxls
import numpy as np


# ---------------------------------------------------------------------------
# Variable: exponential parameters [a, b, c]
#   y = a * exp(b * x) + c
# ---------------------------------------------------------------------------

class ExpVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.array([1.0, 0.5, 0.0]),
):
    """Exponential parameters: [a, b, c]."""


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def _exp_residual_impl(vals, var, xy):
    params = vals[var]
    a, b, c = params[0], params[1], params[2]
    x, y = xy[0], xy[1]
    return jnp.atleast_1d(y - (a * jnp.exp(b * x) + c))


@jaxls.Cost.factory
def exp_residual(
    vals: jaxls.VarValues, var: ExpVar, xy: jax.Array,
) -> jax.Array:
    return _exp_residual_impl(vals, var, xy)


def make_robust_exp_cost(irls_weight_fn):
    @jaxls.Cost.factory(irls_weight_fn=irls_weight_fn)
    def robust_exp_residual(
        vals: jaxls.VarValues, var: ExpVar, xy: jax.Array,
    ) -> jax.Array:
        return _exp_residual_impl(vals, var, xy)

    return robust_exp_residual


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data_homoscedastic(
    n_inliers: int = 200,
    n_outliers: int = 20,
    noise_std: float = 0.2,
    seed: int = 42,
):
    """Homoscedastic noise: sigma_r = noise_std for all points.

    Fixed-delta with sigma=noise_std is the principled choice.
    """
    rng = np.random.RandomState(seed)
    gt = dict(a=2.0, b=0.3, c=1.0)

    x_in = rng.uniform(0, 5, n_inliers)
    y_in = gt["a"] * np.exp(gt["b"] * x_in) + gt["c"]
    y_in += rng.normal(0, noise_std, n_inliers)

    x_out = rng.uniform(0, 5, n_outliers)
    y_out = rng.uniform(-5, 30, n_outliers)

    x_all = np.concatenate([x_in, x_out])
    y_all = np.concatenate([y_in, y_out])
    data = jnp.stack([x_all, y_all], axis=-1)

    gt_params = jnp.array([gt["a"], gt["b"], gt["c"]])
    return data, gt_params, gt, noise_std


def generate_data_heteroscedastic(
    n_inliers: int = 200,
    n_outliers: int = 20,
    base_std: float = 0.1,
    seed: int = 42,
):
    """Heteroscedastic noise: sigma_y(x) = base_std * (1 + x).

    No single fixed sigma works for all points. At x=0, sigma=0.1;
    at x=5, sigma=0.6. Adaptive IRLS can track the effective scale.
    """
    rng = np.random.RandomState(seed)
    gt = dict(a=2.0, b=0.3, c=1.0)

    x_in = rng.uniform(0, 5, n_inliers)
    y_in = gt["a"] * np.exp(gt["b"] * x_in) + gt["c"]
    sigma_per_point = base_std * (1.0 + x_in)
    y_in += rng.normal(0, 1, n_inliers) * sigma_per_point

    x_out = rng.uniform(0, 5, n_outliers)
    y_out = rng.uniform(-5, 30, n_outliers)

    x_all = np.concatenate([x_in, x_out])
    y_all = np.concatenate([y_in, y_out])
    data = jnp.stack([x_all, y_all], axis=-1)

    gt_params = jnp.array([gt["a"], gt["b"], gt["c"]])
    mean_std = float(np.mean(sigma_per_point))
    return data, gt_params, gt, mean_std

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_exp(data, irls_weight_fn=None):
    n = len(data)
    var = ExpVar(id=0)
    init_vals = jaxls.VarValues.make([
        var.with_value(jnp.array([1.8, 0.4, 0.8]))
    ])

    if irls_weight_fn is not None:
        cost_fn = make_robust_exp_cost(irls_weight_fn)
    else:
        cost_fn = exp_residual

    costs = [cost_fn(ExpVar(id=jnp.zeros(n, dtype=jnp.int32)), data)]
    problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
    solution = problem.solve(init_vals, verbose=False)
    return solution[var]


def print_result(label, params, gt_params):
    errs = jnp.abs(params - gt_params)
    print(
        f"  {label:<32} "
        f"a={float(params[0]):>6.3f}  b={float(params[1]):>6.3f}  c={float(params[2]):>6.3f}  "
        f"err(a={float(errs[0]):.4f}, b={float(errs[1]):.4f}, c={float(errs[2]):.4f}) "
        f"total={jnp.linalg.norm(errs):.4f} "
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(data, gt_params, gt, noise_std, n_in, sigmas=None):
    n = len(data)
    n_out = n - n_in

    if sigmas is None:
        sigmas = (noise_std / 2, noise_std, noise_std * 2)

    print(f"Ground truth: a={gt['a']}, b={gt['b']}, c={gt['c']}")
    print(f"Data: {n_in} inliers (true sigma={noise_std:.3f}) + {n_out} outliers\n")

    # ---- Standard LS ----
    print("Standard LS:")
    params = solve_exp(data)
    print_result("  standard", params, gt_params)

    # ---- Fixed-delta Cauchy IRLS ----
    c_cauchy = 2.3849
    print(f"\nFixed-delta Cauchy (c_tuning={c_cauchy}, 95% ARE):")
    for sigma in sigmas:
        c_eff = c_cauchy * sigma
        params = solve_exp(data, jaxls.utils.irls_cauchy(c=c_eff))
        print_result(f"  sigma={sigma:.3f}, c={c_eff:.4f}", params, gt_params)

    # ---- Adaptive Cauchy IRLS ----
    print("\nAdaptive Cauchy:")
    params = solve_exp(data, jaxls.utils.irls_cauchy_adaptive())
    print_result("  adaptive", params, gt_params)

    # ---- Fixed-delta Huber IRLS ----
    c_huber = 1.345
    print(f"\nFixed-delta Huber (c_tuning={c_huber}, 95% ARE):")
    for sigma in sigmas:
        c_eff = c_huber * sigma
        params = solve_exp(data, jaxls.utils.irls_huber(delta=c_eff))
        print_result(f"  sigma={sigma:.3f}, c={c_eff:.4f}", params, gt_params)

    # ---- Adaptive Huber IRLS ----
    print("\nAdaptive Huber:")
    params = solve_exp(data, jaxls.utils.irls_huber_adaptive())
    print_result("  adaptive", params, gt_params)


def main():
    print("=" * 80)
    print("  Scenario 1: Homoscedastic noise (fixed-delta friendly)")
    print("  sigma_r = 0.2 for all points — fixed-delta with true sigma is optimal")
    print("=" * 80)
    data, gt_params, gt, noise_std = generate_data_homoscedastic()
    run_experiment(data, gt_params, gt, noise_std, n_in=200)

    print("\n")
    print("=" * 80)
    print("  Scenario 2: Heteroscedastic noise (adaptive friendly)")
    print("  sigma_y(x) = 0.1*(1+x), ranges from 0.1 to 0.6 — no single sigma works")
    print("=" * 80)
    data, gt_params, gt, mean_std = generate_data_heteroscedastic()
    run_experiment(data, gt_params, gt, mean_std, n_in=200)

    print("\n")
    print("=" * 80)
    print("  Scenario 3: Misspecified sigma (adaptive wins)")
    print("  True sigma=0.2, but user guesses sigma=1.0 (5x too large)")
    print("  Fixed-delta with wrong sigma treats outliers as inliers;")
    print("  adaptive estimates the correct scale from data.")
    print("=" * 80)
    data, gt_params, gt, noise_std = generate_data_homoscedastic()
    wrong_sigma = 1.0
    run_experiment(
        data, gt_params, gt, noise_std, n_in=200,
        sigmas=(wrong_sigma, wrong_sigma * 2, wrong_sigma * 3),
    )


if __name__ == "__main__":
    main()
