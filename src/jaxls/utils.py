import contextlib
import inspect
import time
from functools import partial
from typing import Any, Callable, Generator

import jax
import numpy as onp
import termcolor
from jax import numpy as jnp
from loguru import logger


def tikhonov_floor(dtype: Any) -> float:
    """Precision-adaptive Tikhonov floor for a robust SPD solve of the Schur
    reduced system. Forming S = H_cc - W V^{-1} W^T cancels catastrophically in
    float32 and can leave S numerically indefinite; this floor (added to the
    Jacobi-scaled diagonal) restores positive-definiteness without measurably
    perturbing float64 solves. Single source of truth for both the dense
    on-device path (`_schur._solve_spd_scaled`) and the sparse CHOLMOD host
    path (`_solvers._cholmod_solve_symmetric_on_host`), so the tuned constant
    cannot drift between them. `dtype` may be a JAX or numpy float dtype."""
    eps = float(onp.finfo(dtype).eps)
    return eps * (2e4 if onp.dtype(dtype) == onp.float32 else 4.0)


# Batched products over a tiny contraction axis, written as explicit
# broadcast-multiply-sums rather than `einsum` / `dot_general`. When the
# contraction dimension is tiny (a residual dim of 2, a landmark dim of 3),
# XLA lowers the batched-GEMM form to a kernel that is ~5-30x slower than
# the elementwise form on GPU (and modestly slower on CPU). These products
# dominate Schur-complement assembly and block-Jacobi preconditioner
# construction, so the form matters. Measurements: benchmarks/results.md,
# "Where the GPU time went: batched einsum vs broadcast".


def _batched_gram(a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-row blocks summed over the *middle* (contraction) axis:
    ``a[...,r,i], b[...,r,j] -> out[...,i,j] = sum_r a[...,r,i] b[...,r,j]``."""
    return jnp.sum(a[..., :, :, None] * b[..., :, None, :], axis=-3)


def _batched_outer_last(a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-row blocks summed over the *last* axis:
    ``a[...,t,f], b[...,s,f] -> out[...,t,s] = sum_f a[...,t,f] b[...,s,f]``."""
    return jnp.sum(a[..., :, None, :] * b[..., None, :, :], axis=-1)


def _batched_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Per-row matrix products ``a[n] @ b[n]``:
    ``a[...,t,e], b[...,e,f] -> out[...,t,f] = sum_e a[...,t,e] b[...,e,f]``."""
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


@contextlib.contextmanager
def stopwatch(label: str = "unlabeled block") -> Generator[None, None, None]:
    """Context manager for measuring runtime."""
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")


def _log(fmt: str, *args, **kwargs) -> None:
    logger.bind(function="log").info(fmt, *args, **kwargs)


def jax_log(fmt: str, *args, **kwargs) -> None:
    """Emit a loguru info message from a JITed JAX function."""
    jax.debug.callback(partial(_log, fmt), *args, **kwargs)


def print_deprecation_warning(
    message0: str, message1: str | None = None, stack_level: int = 2
) -> None:
    """Print a nicely formatted deprecation warning with code context.

    Args:
        message0: The deprecation message to display. Goes above code.
        message1: An optional second message to display. Goes under code.
        stack_level: Number of frames to go back to find the caller.
                    (default: 2, meaning the caller of the caller of this function)
    """
    # Get the caller's frame based on stack_level.
    frame = inspect.currentframe()
    for _ in range(stack_level):
        if not frame:
            return
        frame = frame.f_back

    if frame is None:
        return

    # Get more context (2 lines).
    frame_info = inspect.getframeinfo(frame, context=2)
    filename = frame_info.filename
    lineno = frame_info.lineno

    if frame_info.code_context is None:
        return

    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text

    console = Console(stderr=True)
    panel_content = [
        Text.from_markup(message0),
        Syntax("# " + filename + ":" + str(lineno), "python"),
    ]

    # Add code context if available.
    if frame_info.code_context:
        # Get the original code context.
        code_lines = frame_info.code_context
        start_line = lineno - (len(code_lines) // 2)

        while start_line < 1:
            start_line += 1
            code_lines.pop()

        code = "".join(code_lines)

        # Calculate the line number that should be highlighted.
        highlight_line = start_line + (lineno - start_line)

        # Create syntax highlighting with the current line highlighted.
        syntax = Syntax(
            code,
            "python",
            line_numbers=True,
            start_line=start_line,  # Keep the original start line.
            highlight_lines={highlight_line},  # Highlight the actual code line.
        )
        panel_content.append(syntax)

    # Add the second message if provided.
    if message1 is not None:
        panel_content.append(Text.from_markup(message1))

    # Create a group with all content.
    content_group = Group(*panel_content)

    # Create a single panel with all information.
    console.print(
        Panel(
            content_group,
            title="[bold red]Deprecation Warning[/bold red]",
            border_style="red",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# IRLS weight factory functions
# ---------------------------------------------------------------------------

# Consistency constant: for X ~ N(0,1), E[|X|] = median(|X|) / 0.6745.
_MAD_CONSISTENCY_FACTOR = 0.6745


def irls_huber(delta: float = 1.0, eps: float = 1e-10) -> Callable[[jax.Array], jax.Array]:
    """Return a Huber IRLS weight function with a **fixed** scale threshold.

    Produces weights that correspond to the Huber M-estimator loss:

    .. math::

        w_i = \\begin{cases}1 & |r_i| \\le \\delta \\\\ \\delta / |r_i| & |r_i| > \\delta\\end{cases}

    This gives a smooth transition between L2 (small residuals) and L1
    (large residuals) behaviour, making the solver robust to outliers while
    maintaining quadratic convergence near the optimum.

    .. note::

        The threshold ``delta`` is an absolute value in the same units as the
        residuals.  If you want to let the solver estimate the noise scale
        automatically from the data at each iteration, use
        :func:`irls_huber_adaptive` instead.

    Args:
        delta: Threshold that separates the quadratic and linear regimes.
            Residuals with ``|r| <= delta`` receive weight 1; larger residuals
            are down-weighted proportionally.  Default is ``1.0``.
        eps: Small positive constant added to the denominator for numerical
            stability.  Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.  ``residuals`` has shape
        ``(count, residual_flat_dim)``; ``weights`` has the same shape.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            abs_r = jnp.abs(r)
            return jnp.where(abs_r <= delta, jnp.ones_like(abs_r), delta / (abs_r + eps))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_cauchy(c: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """Return a Cauchy (Lorentzian) IRLS weight function with a **fixed** scale.

    Produces weights corresponding to the Cauchy M-estimator loss
    ``\\rho(r) = c^2 / 2 * log(1 + (r/c)^2)``:

    .. math::

        w_i = \\frac{1}{1 + (r_i / c)^2}

    The Cauchy estimator is more aggressive than Huber at down-weighting
    large residuals (sub-linear growth), giving stronger outlier rejection
    but potentially slower convergence.

    .. note::

        ``c`` is an absolute scale value.  See :func:`irls_cauchy_adaptive`
        for automatic scale estimation from the data.

    Args:
        c: Scale parameter.  Residuals much larger than ``c`` are strongly
            down-weighted.  Default is ``1.0``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            return 1.0 / (1.0 + (r / c) ** 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_tukey(c: float = 4.685) -> Callable[[jax.Array], jax.Array]:
    """Return a Tukey bisquare IRLS weight function with a **fixed** scale.

    Produces weights corresponding to the Tukey bisquare M-estimator:

    .. math::

        w_i = \\begin{cases}(1 - (r_i/c)^2)^2 & |r_i| \\le c \\\\ 0 & |r_i| > c\\end{cases}

    Residuals beyond the threshold ``c`` receive *zero* weight and are
    completely ignored.  This gives the strongest outlier rejection of the
    built-in estimators, but can cause instability if the initial estimate is
    poor (convergence to the correct solution is not guaranteed).

    .. note::

        ``c`` is an absolute scale value.  See :func:`irls_tukey_adaptive`
        for automatic scale estimation from the data.

    Args:
        c: Threshold beyond which residuals are ignored.  The default value
            of ``4.685`` gives 95 % efficiency under Gaussian noise when the
            true noise standard deviation equals 1.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            u = r / c
            return jnp.where(jnp.abs(u) <= 1.0, (1.0 - u ** 2) ** 2, jnp.zeros_like(u))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_l1(eps: float = 1e-6) -> Callable[[jax.Array], jax.Array]:
    """Return an L1-norm IRLS weight function.

    Produces weights that convert a least-squares solver into an approximate
    L1 minimiser:

    .. math::

        w_i = \\frac{1}{|r_i| + \\varepsilon}

    Args:
        eps: Small positive constant for numerical stability near zero.
            Default is ``1e-6``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            return 1.0 / (jnp.abs(r) + eps)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_geman_mcclure(c: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """Return a Geman-McClure IRLS weight function with a **fixed** scale.

    Produces weights corresponding to the Geman-McClure M-estimator loss
    ``\\rho(r) = r^2 / (2 (1 + (r/c)^2))``:

    .. math::

        w_i = \\frac{1}{\\bigl(1 + (r_i / c)^2\\bigr)^2}

    The Geman-McClure estimator provides very aggressive outlier rejection —
    even more than Cauchy — because the weight decays as the *square* of the
    Cauchy weight.  It is redescending (the influence function turns back
    toward zero for large residuals).

    Args:
        c: Scale parameter.  Residuals much larger than ``c`` are strongly
            down-weighted.  Default is ``1.0``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            return 1.0 / (1.0 + (r / c) ** 2) ** 2

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_welsh(c: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """Return a Welsh IRLS weight function with a **fixed** scale.

    Produces weights corresponding to the Welsh (Dennis–Welsch) M-estimator
    loss ``\\rho(r) = c^2 / 2 \\cdot (1 - \\exp(-(r/c)^2))``:

    .. math::

        w_i = \\exp\\!\\bigl(-(r_i / c)^2\\bigr)

    The Welsh estimator provides smooth, exponentially decaying weights.
    Like Tukey it is redescending, but it never assigns exactly zero weight,
    which can improve numerical stability.

    Args:
        c: Scale parameter.  Controls how quickly the weight decays with
            residual magnitude.  Default is ``1.0``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            return jnp.exp(-(r / c) ** 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_lp(p: float = 1.2, eps: float = 1e-6) -> Callable[[jax.Array], jax.Array]:
    """Return an L_p-norm IRLS weight function with a **fixed** scale.

    Produces weights that convert a least-squares solver into an approximate
    L_p minimiser for ``0 < p < 2``:

    .. math::

        w_i = |r_i|^{p - 2}

    When ``p = 2`` the weights are all 1 (standard L2).  As ``p`` decreases
    toward 0, large residuals are down-weighted more aggressively.  ``p = 1``
    recovers L1 (median-like) behaviour.

    Args:
        p: Exponent of the L_p norm.  Must satisfy ``0 < p <= 2``.
            Default is ``1.2``.
        eps: Small positive constant added to ``|r|`` for numerical stability
            near zero.  Default is ``1e-6``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        def _per_instance(r: jax.Array) -> jax.Array:
            return (jnp.abs(r) + eps) ** (p - 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


# ---------------------------------------------------------------------------
# Adaptive-scale IRLS weight factories
#
# These factories estimate the noise scale σ from the current residuals at
# every solver iteration using the Median Absolute Deviation (MAD):
#
#     σ = median(|r|) / 0.6745
#
# The 0.6745 factor makes the estimator consistent for Gaussian noise
# (i.e. E[MAD] = 0.6745 σ when r ~ N(0,σ²)).  The scale-normalised
# residual u = r / σ is then used in the weight formula, so the effective
# threshold adapts to the data spread rather than being fixed a priori.
# ---------------------------------------------------------------------------


def irls_huber_adaptive(
    k: float = 1.345, eps: float = 1e-10
) -> Callable[[jax.Array], jax.Array]:
    """Return a Huber IRLS weight function with **adaptive** scale estimation.

    At each solver iteration the noise scale σ is estimated from the current
    residuals using the Median Absolute Deviation (MAD):

    .. math::

        \\hat{\\sigma} = \\frac{\\operatorname{median}(|r|)}{0.6745}

    The weights are then computed using the scale-normalised residuals
    ``u = r / σ``:

    .. math::

        w_i = \\begin{cases}1 & |u_i| \\le k \\\\ k / |u_i| & |u_i| > k\\end{cases}

    The default ``k = 1.345`` gives 95 % asymptotic efficiency relative to
    the ordinary least-squares estimator under Gaussian noise.

    Args:
        k: Tuning constant (multiples of σ).  Default is ``1.345``.
        eps: Small positive constant added to the denominator for numerical
            stability.  Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.  ``residuals`` has shape
        ``(count, residual_flat_dim)``; ``weights`` has the same shape.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        # Estimate sigma from all residuals in the group.
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: jax.Array) -> jax.Array:
            u = jnp.abs(r) / sigma
            return jnp.where(u <= k, jnp.ones_like(u), k / (u + eps))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_cauchy_adaptive(
    k: float = 2.385, eps: float = 1e-10
) -> Callable[[jax.Array], jax.Array]:
    """Return a Cauchy IRLS weight function with **adaptive** scale estimation.

    At each solver iteration the noise scale σ is estimated via MAD:

    .. math::

        \\hat{\\sigma} = \\frac{\\operatorname{median}(|r|)}{0.6745}

    Weights are computed from scale-normalised residuals ``u = r / σ``:

    .. math::

        w_i = \\frac{1}{1 + (u_i / k)^2}

    The default ``k = 2.385`` gives 95 % asymptotic efficiency under
    Gaussian noise.

    Args:
        k: Tuning constant (multiples of σ).  Default is ``2.385``.
        eps: Small constant for numerical stability in σ estimation.
            Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: jax.Array) -> jax.Array:
            u = r / sigma
            return 1.0 / (1.0 + (u / k) ** 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_tukey_adaptive(
    k: float = 4.685, eps: float = 1e-10
) -> Callable[[jax.Array], jax.Array]:
    """Return a Tukey bisquare IRLS weight function with **adaptive** scale estimation.

    At each solver iteration the noise scale σ is estimated via MAD:

    .. math::

        \\hat{\\sigma} = \\frac{\\operatorname{median}(|r|)}{0.6745}

    Weights are computed from scale-normalised residuals ``u = r / σ``:

    .. math::

        w_i = \\begin{cases}(1 - (u_i / k)^2)^2 & |u_i| \\le k \\\\ 0 & |u_i| > k\\end{cases}

    The default ``k = 4.685`` gives 95 % asymptotic efficiency under
    Gaussian noise.  Residuals with ``|r| > k * σ`` are completely excluded.

    Args:
        k: Tuning constant (multiples of σ).  Default is ``4.685``.
        eps: Small constant for numerical stability in σ estimation.
            Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: jax.Array) -> jax.Array:
            u = r / sigma
            t = u / k
            return jnp.where(jnp.abs(t) <= 1.0, (1.0 - t ** 2) ** 2, jnp.zeros_like(t))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_geman_mcclure_adaptive(
    k: float = 3.0, eps: float = 1e-10
) -> Callable[[jax.Array], jax.Array]:
    """Return a Geman-McClure IRLS weight function with **adaptive** scale estimation.

    At each solver iteration the noise scale σ is estimated via MAD:

    .. math::

        \\hat{\\sigma} = \\frac{\\operatorname{median}(|r|)}{0.6745}

    Weights are computed from scale-normalised residuals ``u = r / σ``:

    .. math::

        w_i = \\frac{1}{\\bigl(1 + (u_i / k)^2\\bigr)^2}

    The Geman-McClure weight decays as the *square* of the Cauchy weight,
    giving very aggressive outlier rejection.  The influence function is
    redescending: it turns back toward zero for large residuals.

    Args:
        k: Tuning constant (multiples of σ).  Default is ``3.0``.
        eps: Small constant for numerical stability in σ estimation.
            Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: jax.Array) -> jax.Array:
            u = r / sigma
            return 1.0 / (1.0 + (u / k) ** 2) ** 2

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_welsh_adaptive(
    k: float = 2.985, eps: float = 1e-10
) -> Callable[[jax.Array], jax.Array]:
    """Return a Welsh IRLS weight function with **adaptive** scale estimation.

    At each solver iteration the noise scale σ is estimated via MAD:

    .. math::

        \\hat{\\sigma} = \\frac{\\operatorname{median}(|r|)}{0.6745}

    Weights are computed from scale-normalised residuals ``u = r / σ``:

    .. math::

        w_i = \\exp\\!\\bigl(-(u_i / k)^2\\bigr)

    The Welsh (Dennis–Welsch) estimator provides smooth, exponentially
    decaying weights.  Like Tukey it is redescending, but it never assigns
    exactly zero weight, which can improve numerical stability.

    The default ``k = 2.985`` gives 95 % asymptotic efficiency under
    Gaussian noise.

    Args:
        k: Tuning constant (multiples of σ).  Default is ``2.985``.
        eps: Small constant for numerical stability in σ estimation.
            Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: jax.Array) -> jax.Array:
            u = r / sigma
            return jnp.exp(-(u / k) ** 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_lp_adaptive(
    p: float = 1.2, eps: float = 1e-6
) -> Callable[[jax.Array], jax.Array]:
    """Return an L_p-norm IRLS weight function with **adaptive** scale estimation.

    At each solver iteration the noise scale σ is estimated via MAD:

    .. math::

        \\hat{\\sigma} = \\frac{\\operatorname{median}(|r|)}{0.6745}

    Weights are computed from scale-normalised residuals ``u = r / σ``:

    .. math::

        w_i = |u_i|^{p - 2}

    When ``p = 2`` the weights are all 1 (standard L2).  As ``p`` decreases
    toward 0, large residuals are down-weighted more aggressively.  ``p = 1``
    recovers L1 (median-like) behaviour.

    Args:
        p: Exponent of the L_p norm.  Must satisfy ``0 < p <= 2``.
            Default is ``1.2``.
        eps: Small positive constant added to ``|u|`` for numerical stability
            near zero.  Default is ``1e-6``.

    Returns:
        A callable ``weight_fn(residuals) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residuals: jax.Array) -> jax.Array:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: jax.Array) -> jax.Array:
            u = jnp.abs(r) / sigma
            return (u + eps) ** (p - 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn
