"""Tests for IRLS (Iteratively Reweighted Least Squares) support."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxls


# ---------------------------------------------------------------------------
# Variable type used throughout these tests
# ---------------------------------------------------------------------------


class ScalarVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(())):
    """A scalar optimization variable."""


# ---------------------------------------------------------------------------
# Helper: simple 1-D scalar regression problem
# ---------------------------------------------------------------------------


def _make_scalar_problem(
    observations: jax.Array,
    irls_weight_fn=None,
) -> jaxls.AnalyzedLeastSquaresProblem:
    """Build a problem that estimates a single scalar value from observations."""
    var = ScalarVar(0)
    costs = [
        jaxls.Cost(
            lambda vals, v, obs: jnp.atleast_1d(vals[v] - obs),
            (var, obs),
            irls_weight_fn=irls_weight_fn,
        )
        for obs in observations
    ]
    return jaxls.LeastSquaresProblem(costs, [var]).analyze()


# ---------------------------------------------------------------------------
# Tests for weight factory functions
# ---------------------------------------------------------------------------


def test_irls_huber_weights_shape():
    r = jnp.array([0.0, 0.5, 1.0, 2.0, 5.0])
    w = jaxls.utils.irls_huber(delta=1.0)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_huber_weights_values():
    delta = 1.0
    r = jnp.array([0.5, 1.0, 2.0])
    w = jaxls.utils.irls_huber(delta=delta)(r)
    # |r| <= delta -> weight 1; |r| > delta -> delta / |r|
    expected = jnp.array([1.0, 1.0, delta / 2.0])
    assert jnp.allclose(w, expected, atol=1e-5)


def test_irls_cauchy_weights():
    r = jnp.array([0.0, 1.0, 5.0])
    w = jaxls.utils.irls_cauchy(c=1.0)(r)
    assert w.shape == r.shape
    # Monotonically decreasing with |r|
    assert float(w[0]) >= float(w[1]) >= float(w[2])
    # At r=0, weight should be 1
    assert jnp.allclose(w[0], jnp.array(1.0))


def test_irls_tukey_weights():
    r = jnp.array([0.0, 1.0, 4.685, 5.0, 10.0])
    w = jaxls.utils.irls_tukey(c=4.685)(r)
    assert w.shape == r.shape
    # Residuals beyond c -> zero weight
    assert float(w[4]) == 0.0
    # Residuals within c -> positive weight
    assert float(w[0]) > 0.0
    assert float(w[1]) > 0.0


def test_irls_l1_weights():
    r = jnp.array([0.0, 1.0, 2.0])
    eps = 1e-6
    w = jaxls.utils.irls_l1(eps=eps)(r)
    assert w.shape == r.shape
    expected = 1.0 / (jnp.abs(r) + eps)
    assert jnp.allclose(w, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Correctness: IRLS recovers L2 solution when all weights == 1
# ---------------------------------------------------------------------------


def test_irls_l2_equivalent():
    """With Huber weights (large delta), IRLS should approximate standard L2."""
    observations = jnp.array([1.0, 2.0, 3.0])
    # Large delta -> all weights == 1 -> same as unweighted L2
    problem_irls = _make_scalar_problem(
        observations,
        irls_weight_fn=jaxls.utils.irls_huber(delta=1e6),
    )
    problem_l2 = _make_scalar_problem(observations)

    var = ScalarVar(0)
    vals_irls = problem_irls.solve(verbose=False)
    vals_l2 = problem_l2.solve(verbose=False)

    assert jnp.allclose(vals_irls[var], vals_l2[var], atol=1e-4)


# ---------------------------------------------------------------------------
# Robustness: IRLS with Huber rejects outliers
# ---------------------------------------------------------------------------


def test_irls_huber_robust_to_outliers():
    """IRLS with Huber weights should give a result closer to the inlier mean
    than plain L2 when outliers are present.

    Setup: 9 clean observations near 0.0 and 1 extreme outlier at 100.0.
    - L2 mean ~= 10 (dominated by outlier)
    - IRLS/Huber mean ~= 0 (robust to outlier)
    """
    clean = jnp.zeros(9)
    outlier = jnp.array([100.0])
    observations = jnp.concatenate([clean, outlier])

    var = ScalarVar(0)
    problem_l2 = _make_scalar_problem(observations)
    problem_irls = _make_scalar_problem(
        observations, irls_weight_fn=jaxls.utils.irls_huber(delta=1.0)
    )

    vals_l2 = problem_l2.solve(verbose=False)
    vals_irls = problem_irls.solve(verbose=False)

    l2_estimate = float(vals_l2[var])
    irls_estimate = float(vals_irls[var])

    # L2 is pulled toward the outlier
    assert abs(l2_estimate - 0.0) > 5.0, (
        f"Expected L2 to be pulled toward outlier, got {l2_estimate}"
    )
    # IRLS should be much closer to the true inlier mean
    assert abs(irls_estimate - 0.0) < 1.0, (
        f"Expected IRLS estimate close to 0.0, got {irls_estimate}"
    )


# ---------------------------------------------------------------------------
# Correctness: irls_weight_fn via Cost.factory decorator
# ---------------------------------------------------------------------------


def test_irls_via_factory_decorator():
    """irls_weight_fn should work when supplied to Cost.factory."""
    var = ScalarVar(0)

    @jaxls.Cost.factory(irls_weight_fn=jaxls.utils.irls_cauchy(c=1.0))
    def obs_cost(
        vals: jaxls.VarValues, v: ScalarVar, target: jax.Array
    ) -> jax.Array:
        return jnp.atleast_1d(vals[v] - target)

    clean = jnp.zeros(5)
    outlier = jnp.array([50.0])
    observations = jnp.concatenate([clean, outlier])

    costs = [obs_cost(var, obs) for obs in observations]
    problem = jaxls.LeastSquaresProblem(costs, [var]).analyze()
    vals = problem.solve(verbose=False)

    # Cauchy IRLS should suppress the outlier
    assert abs(float(vals[var]) - 0.0) < 2.0


# ---------------------------------------------------------------------------
# API: irls_weight_fn=None is a no-op
# ---------------------------------------------------------------------------


def test_irls_weight_fn_none_noop():
    """A cost with irls_weight_fn=None should behave identically to plain L2."""
    observations = jnp.array([1.0, 2.0, 3.0])
    var = ScalarVar(0)

    problem_none = _make_scalar_problem(observations, irls_weight_fn=None)
    problem_l2 = _make_scalar_problem(observations)

    vals_none = problem_none.solve(verbose=False)
    vals_l2 = problem_l2.solve(verbose=False)

    assert jnp.allclose(vals_none[var], vals_l2[var], atol=1e-6)
