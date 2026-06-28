"""Robust exponential fitting: IRLS vs GNC.

Fits y = a * exp(b * x) + c to data contaminated with outliers, comparing:
  1. Standard least squares (no robustness)
  2. Fixed-delta IRLS  (requires c = c_tuning * sigma)
  3. Adaptive IRLS      (threshold-free)
  4. GNC-TLS            (Graduated Non-Convexity, Truncated Least Squares)
  5. GNC-GM             (Graduated Non-Convexity, Geman-McClure)

Since r = y - f(x, p), we have dr/dy = 1, so sigma_r = sigma_y exactly.
This makes it the cleanest test of the fixed-scale theory.

Usage:
    python examples/4_robust_curve_fitting.py
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

_DEFAULT_INIT = jnp.array([1.8, 0.4, 0.8])


def solve_exp(data, irls_weight_fn=None, init_params=None):
    n = len(data)
    var = ExpVar(id=0)
    if init_params is None:
        init_params = _DEFAULT_INIT
    init_vals = jaxls.VarValues.make([var.with_value(init_params)])

    if irls_weight_fn is not None:
        cost_fn = make_robust_exp_cost(irls_weight_fn)
    else:
        cost_fn = exp_residual

    costs = [cost_fn(ExpVar(id=jnp.zeros(n, dtype=jnp.int32)), data)]
    problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
    solution = problem.solve(init_vals, verbose=False)
    return solution[var]


def solve_gnc(data, kernel="tls", noise_bound=0.5, init_params=None):
    n = len(data)
    var = ExpVar(id=0)
    if init_params is None:
        init_params = _DEFAULT_INIT
    init_vals = jaxls.VarValues.make([var.with_value(init_params)])
    costs = [exp_residual(ExpVar(id=jnp.zeros(n, dtype=jnp.int32)), data)]
    result = jaxls.gnc_solve(
        costs,
        [var],
        initial_vals=init_vals,
        gnc_config=jaxls.GNCConfig(kernel=kernel, noise_bound=noise_bound),
        verbose=False,
    )
    assert isinstance(result, tuple)
    vals, summary = result
    return vals[var], summary


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


def run_gnc_experiment():
    """Scenario 4: GNC vs IRLS (GM & TLS kernels) with heavy outlier contamination."""
    n_inliers, n_outliers = 200, 40
    noise_std = 0.2
    noise_bound = 0.5  # ~2.5 sigma
    rng = np.random.RandomState(42)
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

    print(f"Ground truth: a={gt['a']}, b={gt['b']}, c={gt['c']}")
    print(f"Data: {n_inliers} inliers (true sigma={noise_std}) + {n_outliers} outliers")
    print(f"noise_bound = {noise_bound} (~{noise_bound/noise_std:.1f} sigma)\n")

    # ---- Standard LS (baseline) ----
    print("Standard LS:")
    params = solve_exp(data)
    print_result("  standard", params, gt_params)

    # ---- TLS kernel comparison ----
    print(f"\n--- TLS (Truncated Least Squares) kernel, c = {noise_bound} ---")

    print(f"\nIRLS-TLS fixed (c={noise_bound}):")
    params = solve_exp(data, jaxls.utils.irls_tls(c=noise_bound))
    print_result("  fixed", params, gt_params)

    print("\nIRLS-TLS adaptive:")
    params = solve_exp(data, jaxls.utils.irls_tls_adaptive())
    print_result("  adaptive", params, gt_params)

    print(f"\nGNC-TLS (noise_bound={noise_bound}):")
    params, summary = solve_gnc(data, kernel="tls", noise_bound=noise_bound)
    n_inliers_found = int(jnp.sum(summary.weights > 0.5))
    print_result(
        f"  {summary.gnc_iterations} iters, {n_inliers_found} inliers",
        params, gt_params,
    )

    # ---- GM kernel comparison ----
    print(f"\n--- Geman-McClure kernel, c = {noise_bound} ---")

    print(f"\nIRLS-GM fixed (c={noise_bound}):")
    params = solve_exp(data, jaxls.utils.irls_geman_mcclure(c=noise_bound))
    print_result("  fixed", params, gt_params)

    print("\nIRLS-GM adaptive:")
    params = solve_exp(data, jaxls.utils.irls_geman_mcclure_adaptive())
    print_result("  adaptive", params, gt_params)

    print(f"\nGNC-GM (noise_bound={noise_bound}):")
    params, summary = solve_gnc(data, kernel="gm", noise_bound=noise_bound)
    print_result(f"  {summary.gnc_iterations} iters", params, gt_params)


def _generate_saturation_data():
    """Generate data with sensor-saturation outliers.

    150 inliers follow the true exponential; 60 outliers cluster near
    y = 12 (simulating a sensor ceiling). Returns data, gt dict, and
    the number of inliers.
    """
    rng = np.random.RandomState(42)
    gt = dict(a=2.0, b=0.3, c=1.0)
    n_in, n_out = 150, 60

    x_in = rng.uniform(0, 5, n_in)
    y_in = gt["a"] * np.exp(gt["b"] * x_in) + gt["c"]
    y_in += rng.normal(0, 0.2, n_in)

    x_out = rng.uniform(0, 5, n_out)
    y_out = 12.0 + rng.normal(0, 1.5, n_out)

    x_all = np.concatenate([x_in, x_out])
    y_all = np.concatenate([y_in, y_out])
    data = jnp.stack([x_all, y_all], axis=-1)
    gt_params = jnp.array([gt["a"], gt["b"], gt["c"]])
    return data, gt_params, gt, n_in, n_out


def run_gnc_saturation_experiment():
    """Scenario 5: Sensor-saturation outliers — GNC vs IRLS.

    60 outliers cluster near y = 12 (sensor ceiling), mixed with 150
    true inliers.  When the initial guess sits near the saturation
    level, IRLS — both fixed and adaptive — locks onto the outlier
    cluster because the outlier residuals look small from that vantage
    point while the true-curve residuals look large and get rejected.

    GNC-TLS avoids this: its first step is a convex relaxation (all
    weights = 1, i.e. standard LS) whose solution is pulled toward
    the 150-point inlier majority, landing closer to the true model.
    Subsequent graduated weight tightening correctly identifies the
    inlier set.
    """
    data, gt_params, gt, n_in, n_out = _generate_saturation_data()
    noise_bound = 1.0

    good_init = jnp.array([1.8, 0.4, 0.8])
    bad_init = jnp.array([0.5, 0.1, 10.0])

    print(f"Ground truth: a={gt['a']}, b={gt['b']}, c={gt['c']}")
    print(f"Data: {n_in} inliers + {n_out} outliers (clustered near y=12)")
    print(f"noise_bound = {noise_bound}\n")

    for label, init in [("Good init", good_init), ("Bad init (near saturation)", bad_init)]:
        print(f"=== {label}: a={float(init[0]):.1f}, b={float(init[1]):.2f}, c={float(init[2]):.1f} ===\n")

        print("Standard LS:")
        params = solve_exp(data, init_params=init)
        print_result("  standard", params, gt_params)

        print(f"\n--- TLS kernel, c = {noise_bound} ---")

        print(f"\nIRLS-TLS fixed (c={noise_bound}):")
        params = solve_exp(data, jaxls.utils.irls_tls(c=noise_bound), init_params=init)
        print_result("  fixed", params, gt_params)

        print("\nIRLS-TLS adaptive:")
        params = solve_exp(data, jaxls.utils.irls_tls_adaptive(), init_params=init)
        print_result("  adaptive", params, gt_params)

        print(f"\nGNC-TLS (noise_bound={noise_bound}):")
        params, summary = solve_gnc(data, kernel="tls", noise_bound=noise_bound, init_params=init)
        n_inliers_found = int(jnp.sum(summary.weights > 0.5))
        print_result(
            f"  {summary.gnc_iterations} iters, {n_inliers_found} inliers",
            params, gt_params,
        )

        print(f"\n--- Geman-McClure kernel, c = {noise_bound} ---")

        print(f"\nIRLS-GM fixed (c={noise_bound}):")
        params = solve_exp(data, jaxls.utils.irls_geman_mcclure(c=noise_bound), init_params=init)
        print_result("  fixed", params, gt_params)

        print("\nIRLS-GM adaptive:")
        params = solve_exp(data, jaxls.utils.irls_geman_mcclure_adaptive(), init_params=init)
        print_result("  adaptive", params, gt_params)

        print(f"\nGNC-TLS (noise_bound={noise_bound}):")
        params, summary = solve_gnc(data, kernel="tls", noise_bound=noise_bound, init_params=init)
        print_result(f"  {summary.gnc_iterations} iters", params, gt_params)

        print()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _exp_curve(params, x):
    """Evaluate y = a*exp(b*x) + c."""
    a, b, c = float(params[0]), float(params[1]), float(params[2])
    return a * np.exp(b * x) + c


def plot_all_scenarios():
    import matplotlib.pyplot as plt

    x_dense = np.linspace(0, 5, 200)
    gt_params_arr = np.array([2.0, 0.3, 1.0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ================================================================
    # Row 1: Scenarios 1-3  (IRLS focus)
    # ================================================================

    # -- Scenario 1: Homoscedastic --
    data, gt_params, gt, noise_std = generate_data_homoscedastic()
    x_data, y_data = np.asarray(data[:, 0]), np.asarray(data[:, 1])
    n_in = 200

    default_init = np.asarray(_DEFAULT_INIT)
    init_style = dict(color="gray", lw=1.5, ls=":", alpha=0.7)

    ax = axes[0, 0]
    ax.scatter(x_data[:n_in], y_data[:n_in], s=8, alpha=0.4, c="tab:blue", label="Inliers")
    ax.scatter(x_data[n_in:], y_data[n_in:], s=20, marker="x", c="tab:red", label="Outliers")
    ax.plot(x_dense, _exp_curve(gt_params_arr, x_dense), "k--", lw=2, label="Ground truth")
    ax.plot(x_dense, _exp_curve(default_init, x_dense), label="Initial guess", **init_style)

    for label, wfn in [
        ("Standard LS", None),
        (f"Cauchy fixed (c={2.3849*noise_std:.2f})", jaxls.utils.irls_cauchy(c=2.3849 * noise_std)),
        ("Cauchy adaptive", jaxls.utils.irls_cauchy_adaptive()),
    ]:
        p = solve_exp(data, wfn)
        ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, label=label)

    ax.set_title("Scenario 1: Homoscedastic noise")
    ax.set_ylim(-6, 32)
    ax.legend(fontsize=7, loc="upper left")

    # -- Scenario 2: Heteroscedastic --
    data, gt_params, gt, mean_std = generate_data_heteroscedastic()
    x_data, y_data = np.asarray(data[:, 0]), np.asarray(data[:, 1])

    ax = axes[0, 1]
    ax.scatter(x_data[:n_in], y_data[:n_in], s=8, alpha=0.4, c="tab:blue", label="Inliers")
    ax.scatter(x_data[n_in:], y_data[n_in:], s=20, marker="x", c="tab:red", label="Outliers")
    ax.plot(x_dense, _exp_curve(gt_params_arr, x_dense), "k--", lw=2, label="Ground truth")
    ax.plot(x_dense, _exp_curve(default_init, x_dense), label="Initial guess", **init_style)

    for label, wfn in [
        ("Standard LS", None),
        (f"Cauchy fixed (c={2.3849*mean_std:.2f})", jaxls.utils.irls_cauchy(c=2.3849 * mean_std)),
        ("Cauchy adaptive", jaxls.utils.irls_cauchy_adaptive()),
    ]:
        p = solve_exp(data, wfn)
        ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, label=label)

    ax.set_title("Scenario 2: Heteroscedastic noise")
    ax.set_ylim(-6, 32)
    ax.legend(fontsize=7, loc="upper left")

    # -- Scenario 3: Misspecified sigma --
    data, gt_params, gt, noise_std = generate_data_homoscedastic()
    x_data, y_data = np.asarray(data[:, 0]), np.asarray(data[:, 1])
    wrong_sigma = 1.0

    ax = axes[0, 2]
    ax.scatter(x_data[:n_in], y_data[:n_in], s=8, alpha=0.4, c="tab:blue", label="Inliers")
    ax.scatter(x_data[n_in:], y_data[n_in:], s=20, marker="x", c="tab:red", label="Outliers")
    ax.plot(x_dense, _exp_curve(gt_params_arr, x_dense), "k--", lw=2, label="Ground truth")
    ax.plot(x_dense, _exp_curve(default_init, x_dense), label="Initial guess", **init_style)

    for label, wfn in [
        ("Standard LS", None),
        (f"Cauchy fixed (c={2.3849*wrong_sigma:.2f}, wrong!)", jaxls.utils.irls_cauchy(c=2.3849 * wrong_sigma)),
        ("Cauchy adaptive", jaxls.utils.irls_cauchy_adaptive()),
    ]:
        p = solve_exp(data, wfn)
        ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, label=label)

    ax.set_title("Scenario 3: Misspecified sigma")
    ax.set_ylim(-6, 32)
    ax.legend(fontsize=7, loc="upper left")

    # ================================================================
    # Row 2: Scenarios 4-5  (GNC focus)
    # ================================================================

    # -- Scenario 4: GNC vs IRLS, random outliers --
    n_in4, n_out4 = 200, 40
    noise_std4 = 0.2
    noise_bound4 = 0.5
    rng = np.random.RandomState(42)
    x_in = rng.uniform(0, 5, n_in4)
    y_in = 2.0 * np.exp(0.3 * x_in) + 1.0 + rng.normal(0, noise_std4, n_in4)
    x_out = rng.uniform(0, 5, n_out4)
    y_out = rng.uniform(-5, 30, n_out4)
    data4 = jnp.stack([np.concatenate([x_in, x_out]),
                        np.concatenate([y_in, y_out])], axis=-1)
    x_data4, y_data4 = np.asarray(data4[:, 0]), np.asarray(data4[:, 1])

    ax = axes[1, 0]
    ax.scatter(x_data4[:n_in4], y_data4[:n_in4], s=8, alpha=0.4, c="tab:blue", label="Inliers")
    ax.scatter(x_data4[n_in4:], y_data4[n_in4:], s=20, marker="x", c="tab:red", label="Outliers")
    ax.plot(x_dense, _exp_curve(gt_params_arr, x_dense), "k--", lw=2, label="Ground truth")
    ax.plot(x_dense, _exp_curve(default_init, x_dense), label="Initial guess", **init_style)

    p = solve_exp(data4)
    ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, label="Standard LS")
    p = solve_exp(data4, jaxls.utils.irls_tls(c=noise_bound4))
    ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, label=f"IRLS-TLS (c={noise_bound4})")
    p = solve_exp(data4, jaxls.utils.irls_geman_mcclure(c=noise_bound4))
    ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, label=f"IRLS-GM (c={noise_bound4})")
    p, _ = solve_gnc(data4, kernel="tls", noise_bound=noise_bound4)
    ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, ls="-.", label="GNC-TLS")
    p, _ = solve_gnc(data4, kernel="gm", noise_bound=noise_bound4)
    ax.plot(x_dense, _exp_curve(p, x_dense), lw=1.5, ls="-.", label="GNC-GM")

    ax.set_title("Scenario 4: GNC vs IRLS (random outliers)")
    ax.set_ylim(-6, 32)
    ax.legend(fontsize=7, loc="upper left")

    # -- Scenario 5: Sensor-saturation outliers --
    data5, _, _, n_in5, n_out5 = _generate_saturation_data()
    x_data5, y_data5 = np.asarray(data5[:, 0]), np.asarray(data5[:, 1])
    noise_bound5 = 1.0

    good_init = jnp.array([1.8, 0.4, 0.8])
    bad_init = jnp.array([0.5, 0.1, 10.0])

    for col, (init_label, init) in enumerate([
        ("Good init", good_init),
        ("Bad init (near saturation)", bad_init),
    ]):
        ax = axes[1, 1 + col]
        ax.scatter(x_data5[:n_in5], y_data5[:n_in5], s=8, alpha=0.4, c="tab:blue", label="Inliers")
        ax.scatter(x_data5[n_in5:], y_data5[n_in5:], s=20, marker="x", c="tab:red",
                   alpha=0.5, label="Saturated outliers")
        ax.plot(x_dense, _exp_curve(gt_params_arr, x_dense), "k--", lw=2, label="Ground truth")
        ax.plot(x_dense, _exp_curve(np.asarray(init), x_dense), label="Initial guess", **init_style)

        methods = [
            ("Standard LS",              dict(lw=1.2, ls="-", alpha=0.7),
             solve_exp(data5, init_params=init)),
            (f"IRLS-TLS (c={noise_bound5})", dict(lw=1.5, ls="-"),
             solve_exp(data5, jaxls.utils.irls_tls(c=noise_bound5), init_params=init)),
            ("IRLS-TLS adaptive",        dict(lw=1.5, ls="-"),
             solve_exp(data5, jaxls.utils.irls_tls_adaptive(), init_params=init)),
            (f"IRLS-GM (c={noise_bound5})",  dict(lw=1.5, ls="-"),
             solve_exp(data5, jaxls.utils.irls_geman_mcclure(c=noise_bound5), init_params=init)),
            ("IRLS-GM adaptive",         dict(lw=1.5, ls="-"),
             solve_exp(data5, jaxls.utils.irls_geman_mcclure_adaptive(), init_params=init)),
            ("GNC-TLS",                  dict(lw=2.5, ls="-."),
             solve_gnc(data5, kernel="tls", noise_bound=noise_bound5, init_params=init)[0]),
            ("GNC-GM",                   dict(lw=2.5, ls="-."),
             solve_gnc(data5, kernel="gm", noise_bound=noise_bound5, init_params=init)[0]),
        ]
        for name, style, p in methods:
            y_fit = _exp_curve(p, x_dense)
            if np.max(np.abs(y_fit)) < 50:
                ax.plot(x_dense, y_fit, label=name, **style)
            else:
                ax.plot([], [], label=f"{name} (diverged)", **style)

        ax.set_title(f"Scenario 5: {init_label}")
        ax.set_ylim(-5, 20)
        ax.legend(fontsize=6.5, loc="upper left")

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Robust Exponential Fitting: IRLS vs GNC", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig("examples/robust_curve_fitting.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to examples/robust_curve_fitting.png")
    plt.show()


if __name__ == "__main__":
    main()

    print("\n")
    print("=" * 80)
    print("  Scenario 4: GNC vs IRLS — same kernel, same threshold")
    print("  Compares GNC outer loop vs plain IRLS for TLS and GM kernels,")
    print("  with both fixed-scale (c = noise_bound) and adaptive variants.")
    print("=" * 80)
    run_gnc_experiment()

    print("\n")
    print("=" * 80)
    print("  Scenario 5: GNC vs IRLS — sensor-saturation outliers")
    print("  Outliers cluster near y=12 (sensor ceiling).")
    print("  Bad init near saturation traps IRLS; GNC escapes via convex relaxation.")
    print("=" * 80)
    run_gnc_saturation_experiment()

    print("\n")
    print("=" * 80)
    print("  Plotting all scenarios...")
    print("=" * 80)
    plot_all_scenarios()
