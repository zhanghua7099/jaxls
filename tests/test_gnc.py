"""Tests for GNC (Graduated Non-Convexity) robust solver."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxls


class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(())):
    """A scalar optimization variable."""


def _make_gnc_problem(
    observations: jax.Array,
    noise_bound: float = 1.0,
    kernel: str = "tls",
):
    """Build a scalar estimation problem and solve with GNC."""
    var = ScalarVar(0)
    costs = [
        jaxls.Cost(
            lambda vals, v, obs: jnp.atleast_1d(vals[v] - obs),
            (var, obs),
        )
        for obs in observations
    ]
    result = jaxls.gnc_solve(
        costs,
        [var],
        gnc_config=jaxls.GNCConfig(
            kernel=kernel,
            noise_bound=noise_bound,
        ),
        verbose=False,
    )
    assert isinstance(result, tuple)
    return result


def test_gnc_tls_rejects_outliers():
    """GNC-TLS should classify outliers and give a result close to the inlier mean."""
    clean = jnp.zeros(9)
    outlier = jnp.array([100.0])
    observations = jnp.concatenate([clean, outlier])

    vals, summary = _make_gnc_problem(observations, noise_bound=1.0, kernel="tls")
    estimate = float(vals[ScalarVar(0)])
    assert abs(estimate) < 1.0, f"Expected GNC-TLS estimate near 0, got {estimate}"
    assert summary.gnc_iterations > 0


def test_gnc_gm_rejects_outliers():
    """GNC-GM should down-weight outliers."""
    clean = jnp.zeros(9)
    outlier = jnp.array([50.0])
    observations = jnp.concatenate([clean, outlier])

    vals, summary = _make_gnc_problem(observations, noise_bound=1.0, kernel="gm")
    estimate = float(vals[ScalarVar(0)])
    assert abs(estimate) < 2.0, f"Expected GNC-GM estimate near 0, got {estimate}"


def test_gnc_no_outliers():
    """With no outliers, GNC should recover the mean (same as L2)."""
    observations = jnp.array([1.0, 2.0, 3.0])
    vals, summary = _make_gnc_problem(observations, noise_bound=5.0, kernel="tls")
    estimate = float(vals[ScalarVar(0)])
    assert abs(estimate - 2.0) < 0.1, f"Expected mean ~2.0, got {estimate}"


def test_gnc_weights_binary_tls():
    """GNC-TLS should converge to binary weights."""
    clean = jnp.ones(8)
    outliers = jnp.array([50.0, -30.0])
    observations = jnp.concatenate([clean, outliers])

    _, summary = _make_gnc_problem(observations, noise_bound=2.0, kernel="tls")
    w = np.asarray(summary.weights)
    assert np.all((w < 0.01) | (w > 0.99)), "TLS weights should be approximately binary"


def test_gnc_returns_vals_only():
    """return_summary=False should return only VarValues."""
    var = ScalarVar(0)
    costs = [
        jaxls.Cost(
            lambda vals, v, obs: jnp.atleast_1d(vals[v] - obs),
            (var, obs),
        )
        for obs in jnp.array([1.0, 2.0, 3.0])
    ]
    result = jaxls.gnc_solve(
        costs,
        [var],
        gnc_config=jaxls.GNCConfig(noise_bound=5.0),
        verbose=False,
        return_summary=False,
    )
    assert isinstance(result, jaxls.VarValues)


def test_gnc_with_prior_costs():
    """Costs with existing irls_weight_fn should be left unchanged by GNC."""
    var = ScalarVar(0)
    obs_costs = [
        jaxls.Cost(
            lambda vals, v, obs: jnp.atleast_1d(vals[v] - obs),
            (var, obs),
        )
        for obs in jnp.concatenate([jnp.zeros(8), jnp.array([100.0])])
    ]
    prior_cost = jaxls.Cost(
        lambda vals, v: jnp.atleast_1d(vals[v]),
        (var,),
        irls_weight_fn=jaxls.utils.irls_huber(delta=10.0),
    )
    all_costs = obs_costs + [prior_cost]

    result = jaxls.gnc_solve(
        all_costs,
        [var],
        gnc_config=jaxls.GNCConfig(noise_bound=1.0, kernel="tls"),
        verbose=False,
    )
    assert isinstance(result, tuple)
    vals, summary = result
    estimate = float(vals[ScalarVar(0)])
    assert abs(estimate) < 2.0, f"Expected estimate near 0, got {estimate}"


def test_gnc_2d_regression():
    """GNC on a 2D linear regression with outliers."""

    class LineVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        """Line parameters [slope, intercept]."""

    rng = np.random.RandomState(42)
    n_inliers, n_outliers = 50, 10
    x_in = rng.uniform(0, 5, n_inliers)
    y_in = 2.0 * x_in + 1.0 + rng.normal(0, 0.1, n_inliers)
    x_out = rng.uniform(0, 5, n_outliers)
    y_out = rng.uniform(-10, 20, n_outliers)

    x_all = np.concatenate([x_in, x_out])
    y_all = np.concatenate([y_in, y_out])
    data = jnp.stack([x_all, y_all], axis=-1)

    var = LineVar(0)

    @jaxls.Cost.factory
    def line_residual(vals: jaxls.VarValues, v: LineVar, xy: jax.Array) -> jax.Array:
        params = vals[v]
        return jnp.atleast_1d(xy[1] - (params[0] * xy[0] + params[1]))

    n = len(data)
    costs = [line_residual(LineVar(id=jnp.zeros(n, dtype=jnp.int32)), data)]

    init_vals = jaxls.VarValues.make([var.with_value(jnp.array([1.0, 0.0]))])
    result = jaxls.gnc_solve(
        costs,
        [var],
        initial_vals=init_vals,
        gnc_config=jaxls.GNCConfig(noise_bound=0.3, kernel="tls"),
        verbose=False,
    )
    assert isinstance(result, tuple)
    vals, summary = result
    params = vals[var]
    assert abs(float(params[0]) - 2.0) < 0.3, f"slope error: {float(params[0])}"
    assert abs(float(params[1]) - 1.0) < 0.5, f"intercept error: {float(params[1])}"


def test_gnc_multidim_residual():
    """GNC with multi-dimensional residuals uses per-measurement ‖r‖²."""

    class PointVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(2)):
        """2D point to estimate."""

    rng = np.random.RandomState(0)
    gt = jnp.array([3.0, 4.0])

    # 15 inliers near the true point, 5 outliers far away
    inlier_obs = gt + jnp.array(rng.normal(0, 0.1, (15, 2)))
    outlier_obs = jnp.array(rng.uniform(10, 20, (5, 2)))
    observations = jnp.concatenate([inlier_obs, outlier_obs], axis=0)

    var = PointVar(0)

    # 2D residual per measurement
    @jaxls.Cost.factory
    def point_residual(
        vals: jaxls.VarValues, v: PointVar, obs: jax.Array,
    ) -> jax.Array:
        return vals[v] - obs  # shape (2,)

    costs = [point_residual(var, obs) for obs in observations]
    result = jaxls.gnc_solve(
        costs,
        [var],
        gnc_config=jaxls.GNCConfig(noise_bound=0.5, kernel="tls"),
        verbose=False,
    )
    assert isinstance(result, tuple)
    vals, summary = result
    estimate = vals[var]
    err = float(jnp.linalg.norm(estimate - gt))
    assert err < 0.3, f"Expected estimate near [3, 4], got {estimate}, err={err}"
    # Weights should be per-measurement (20 measurements), not per-element (40)
    assert len(summary.weights) == 20, (
        f"Expected 20 per-measurement weights, got {len(summary.weights)}"
    )
