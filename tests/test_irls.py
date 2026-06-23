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
# Tests for fixed-scale weight factory functions
# Weight functions now expect 2D input: (count, residual_flat_dim)
# ---------------------------------------------------------------------------


def test_irls_huber_weights_shape():
    # (count=2, residual_dim=3)
    r = jnp.array([[0.0, 0.5, 1.0], [2.0, 5.0, 0.2]])
    w = jaxls.utils.irls_huber(delta=1.0)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_huber_weights_values():
    delta = 1.0
    # Each row is one cost instance.
    r = jnp.array([[0.5, 1.0, 2.0]])  # (1, 3)
    w = jaxls.utils.irls_huber(delta=delta)(r)
    # |r| <= delta -> weight 1; |r| > delta -> delta / |r|
    expected = jnp.array([[1.0, 1.0, delta / 2.0]])
    assert jnp.allclose(w, expected, atol=1e-5)


def test_irls_cauchy_weights():
    r = jnp.array([[0.0, 1.0, 5.0]])  # (1, 3)
    w = jaxls.utils.irls_cauchy(c=1.0)(r)
    assert w.shape == r.shape
    # Monotonically decreasing with |r|
    assert float(w[0, 0]) >= float(w[0, 1]) >= float(w[0, 2])
    # At r=0, weight should be 1
    assert jnp.allclose(w[0, 0], jnp.array(1.0))


def test_irls_tukey_weights():
    r = jnp.array([[0.0, 1.0, 4.685, 5.0, 10.0]])  # (1, 5)
    w = jaxls.utils.irls_tukey(c=4.685)(r)
    assert w.shape == r.shape
    # Residuals beyond c -> zero weight
    assert float(w[0, 4]) == 0.0
    # Residuals within c -> positive weight
    assert float(w[0, 0]) > 0.0
    assert float(w[0, 1]) > 0.0


def test_irls_l1_weights():
    r = jnp.array([[0.0, 1.0, 2.0]])  # (1, 3)
    eps = 1e-6
    w = jaxls.utils.irls_l1(eps=eps)(r)
    assert w.shape == r.shape
    expected = 1.0 / (jnp.abs(r) + eps)
    assert jnp.allclose(w, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Tests for adaptive-scale weight factory functions
# ---------------------------------------------------------------------------


def test_irls_huber_adaptive_weights_shape():
    r = jnp.array([[0.0, 0.5, 2.0], [1.0, 3.0, 0.1]])  # (2, 3)
    w = jaxls.utils.irls_huber_adaptive(k=1.345)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_huber_adaptive_scale_normalisation():
    """Scale-normalised Huber: small residuals -> weight 1, large -> down-weighted."""
    # 9 inliers at ±1 (sigma ~ 1), one outlier at 100
    inliers = jnp.ones((9, 1))  # residuals of magnitude 1
    outlier = jnp.array([[100.0]])
    r = jnp.concatenate([inliers, outlier], axis=0)  # (10, 1)
    w = jaxls.utils.irls_huber_adaptive(k=1.345)(r)
    # Inliers should have weight 1 (within k*sigma)
    assert jnp.allclose(w[:9], jnp.ones((9, 1)), atol=0.1)
    # Outlier should be heavily down-weighted
    assert float(w[9, 0]) < 0.1


def test_irls_cauchy_adaptive_weights_shape():
    r = jnp.array([[0.0, 1.0, 5.0]])  # (1, 3)
    w = jaxls.utils.irls_cauchy_adaptive(k=2.385)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_tukey_adaptive_weights_shape():
    r = jnp.array([[0.0, 1.0, 5.0, 10.0]])  # (1, 4)
    w = jaxls.utils.irls_tukey_adaptive(k=4.685)(r)
    assert w.shape == r.shape
    assert jnp.all(w >= 0)


def test_irls_tukey_adaptive_excludes_outliers():
    """Adaptive Tukey should give zero weight to large outliers."""
    # Inliers near 0, outlier very far
    inliers = jnp.zeros((9, 1))
    outlier = jnp.array([[1000.0]])
    r = jnp.concatenate([inliers, outlier], axis=0)  # (10, 1)
    w = jaxls.utils.irls_tukey_adaptive(k=4.685)(r)
    assert float(w[9, 0]) == 0.0


def test_irls_adaptive_sigma_changes_with_scale():
    """Adaptive estimator should give same relative weights regardless of absolute scale."""
    # Same relative outlier-to-inlier ratio, but different absolute scales
    r_small = jnp.array([[0.01, 0.01, 0.01, 0.01, 1.0]])  # (1, 5) scale ~0.01
    r_large = r_small * 100.0  # scale ~1.0

    w_small = jaxls.utils.irls_huber_adaptive()(r_small)
    w_large = jaxls.utils.irls_huber_adaptive()(r_large)

    # The pattern of weights should be the same (last element is outlier in both)
    assert jnp.allclose(w_small, w_large, atol=1e-4), (
        "Adaptive weights should be scale-invariant"
    )


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
# Robustness: fixed-scale IRLS with Huber rejects outliers
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
# Robustness: adaptive-scale IRLS works without specifying thresholds
# ---------------------------------------------------------------------------


def test_irls_huber_adaptive_robust_to_outliers():
    """Adaptive Huber IRLS should reject outliers without requiring a manual threshold."""
    clean = jnp.zeros(9)
    outlier = jnp.array([100.0])
    observations = jnp.concatenate([clean, outlier])

    var = ScalarVar(0)
    problem_irls = _make_scalar_problem(
        observations, irls_weight_fn=jaxls.utils.irls_huber_adaptive()
    )
    vals_irls = problem_irls.solve(verbose=False)
    irls_estimate = float(vals_irls[var])

    assert abs(irls_estimate - 0.0) < 1.0, (
        f"Expected adaptive IRLS estimate close to 0.0, got {irls_estimate}"
    )


def test_irls_cauchy_adaptive_robust_to_outliers():
    """Adaptive Cauchy IRLS should also suppress large outliers."""
    clean = jnp.zeros(9)
    outlier = jnp.array([50.0])
    observations = jnp.concatenate([clean, outlier])

    var = ScalarVar(0)
    problem_irls = _make_scalar_problem(
        observations, irls_weight_fn=jaxls.utils.irls_cauchy_adaptive()
    )
    vals_irls = problem_irls.solve(verbose=False)
    irls_estimate = float(vals_irls[var])

    assert abs(irls_estimate - 0.0) < 2.0, (
        f"Expected adaptive Cauchy IRLS estimate close to 0.0, got {irls_estimate}"
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


# ---------------------------------------------------------------------------
# Tests for Geman-McClure weight functions
# ---------------------------------------------------------------------------


def test_irls_geman_mcclure_weights():
    r = jnp.array([[0.0, 1.0, 5.0]])  # (1, 3)
    w = jaxls.utils.irls_geman_mcclure(c=1.0)(r)
    assert w.shape == r.shape
    # Monotonically decreasing with |r|
    assert float(w[0, 0]) >= float(w[0, 1]) >= float(w[0, 2])
    # At r=0, weight should be 1
    assert jnp.allclose(w[0, 0], jnp.array(1.0))
    # Exact value: 1/(1+r^2)^2
    expected_at_1 = 1.0 / (1.0 + 1.0) ** 2  # 0.25
    assert jnp.allclose(w[0, 1], jnp.array(expected_at_1), atol=1e-5)


def test_irls_geman_mcclure_adaptive_weights_shape():
    r = jnp.array([[0.0, 0.5, 2.0], [1.0, 3.0, 0.1]])  # (2, 3)
    w = jaxls.utils.irls_geman_mcclure_adaptive(k=3.0)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_geman_mcclure_adaptive_robust_to_outliers():
    """Adaptive Geman-McClure should reject outliers."""
    clean = jnp.zeros(9)
    outlier = jnp.array([100.0])
    observations = jnp.concatenate([clean, outlier])

    var = ScalarVar(0)
    problem_irls = _make_scalar_problem(
        observations, irls_weight_fn=jaxls.utils.irls_geman_mcclure_adaptive()
    )
    vals_irls = problem_irls.solve(verbose=False)
    irls_estimate = float(vals_irls[var])

    assert abs(irls_estimate - 0.0) < 1.0, (
        f"Expected adaptive Geman-McClure IRLS estimate close to 0.0, got {irls_estimate}"
    )


# ---------------------------------------------------------------------------
# Tests for Welsh weight functions
# ---------------------------------------------------------------------------


def test_irls_welsh_weights():
    r = jnp.array([[0.0, 1.0, 5.0]])  # (1, 3)
    w = jaxls.utils.irls_welsh(c=1.0)(r)
    assert w.shape == r.shape
    # Monotonically decreasing with |r|
    assert float(w[0, 0]) >= float(w[0, 1]) >= float(w[0, 2])
    # At r=0, weight should be 1
    assert jnp.allclose(w[0, 0], jnp.array(1.0))
    # Exact value: exp(-r^2)
    expected_at_1 = float(jnp.exp(jnp.array(-1.0)))
    assert jnp.allclose(w[0, 1], jnp.array(expected_at_1), atol=1e-5)


def test_irls_welsh_adaptive_weights_shape():
    r = jnp.array([[0.0, 0.5, 2.0], [1.0, 3.0, 0.1]])  # (2, 3)
    w = jaxls.utils.irls_welsh_adaptive(k=2.985)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_welsh_adaptive_robust_to_outliers():
    """Adaptive Welsh should reject outliers."""
    clean = jnp.zeros(9)
    outlier = jnp.array([100.0])
    observations = jnp.concatenate([clean, outlier])

    var = ScalarVar(0)
    problem_irls = _make_scalar_problem(
        observations, irls_weight_fn=jaxls.utils.irls_welsh_adaptive()
    )
    vals_irls = problem_irls.solve(verbose=False)
    irls_estimate = float(vals_irls[var])

    assert abs(irls_estimate - 0.0) < 1.0, (
        f"Expected adaptive Welsh IRLS estimate close to 0.0, got {irls_estimate}"
    )


# ---------------------------------------------------------------------------
# Tests for L_p weight functions
# ---------------------------------------------------------------------------


def test_irls_lp_weights():
    r = jnp.array([[0.5, 1.0, 2.0]])  # (1, 3)
    p = 1.2
    eps = 1e-6
    w = jaxls.utils.irls_lp(p=p, eps=eps)(r)
    assert w.shape == r.shape
    # For p < 2, larger residuals get smaller weights
    assert float(w[0, 0]) > float(w[0, 1]) > float(w[0, 2])


def test_irls_lp_p2_equivalent():
    """With p=2, L_p weights should all be ~1, recovering standard L2."""
    observations = jnp.array([1.0, 2.0, 3.0])
    problem_lp = _make_scalar_problem(
        observations,
        irls_weight_fn=jaxls.utils.irls_lp(p=2.0),
    )
    problem_l2 = _make_scalar_problem(observations)

    var = ScalarVar(0)
    vals_lp = problem_lp.solve(verbose=False)
    vals_l2 = problem_l2.solve(verbose=False)

    assert jnp.allclose(vals_lp[var], vals_l2[var], atol=1e-3)


def test_irls_lp_adaptive_weights_shape():
    r = jnp.array([[0.0, 0.5, 2.0], [1.0, 3.0, 0.1]])  # (2, 3)
    w = jaxls.utils.irls_lp_adaptive(p=1.2)(r)
    assert w.shape == r.shape
    assert jnp.all(w > 0)


def test_irls_lp_adaptive_robust_to_outliers():
    """Adaptive L_p (p=1.0) should approximate L1 and reject outliers."""
    clean = jnp.zeros(9)
    outlier = jnp.array([100.0])
    observations = jnp.concatenate([clean, outlier])

    var = ScalarVar(0)
    problem_irls = _make_scalar_problem(
        observations, irls_weight_fn=jaxls.utils.irls_lp_adaptive(p=1.0)
    )
    vals_irls = problem_irls.solve(verbose=False)
    irls_estimate = float(vals_irls[var])

    assert abs(irls_estimate - 0.0) < 1.0, (
        f"Expected adaptive L_p IRLS estimate close to 0.0, got {irls_estimate}"
    )


def test_irls_adaptive_scale_invariance_all():
    """All adaptive estimators should give same relative weights regardless of scale."""
    r_small = jnp.array([[0.01, 0.01, 0.01, 0.01, 1.0]])
    r_large = r_small * 100.0

    for name, factory in [
        ("geman_mcclure", jaxls.utils.irls_geman_mcclure_adaptive),
        ("welsh", jaxls.utils.irls_welsh_adaptive),
        ("lp", lambda: jaxls.utils.irls_lp_adaptive(p=1.2)),
    ]:
        fn = factory() if callable(factory) else factory
        w_small = fn(r_small)
        w_large = fn(r_large)
        assert jnp.allclose(w_small, w_large, atol=1e-3), (
            f"Adaptive {name} weights should be scale-invariant"
        )
