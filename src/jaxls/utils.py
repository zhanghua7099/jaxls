import contextlib
import inspect
import time
from functools import partial
from typing import Callable, Generator

import jax
import termcolor
from jax import numpy as jnp
from loguru import logger


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


def irls_huber(delta: float = 1.0, eps: float = 1e-10) -> Callable[[jax.Array], jax.Array]:
    """Return a Huber IRLS weight function.

    Produces weights that correspond to the Huber M-estimator loss:

    .. math::

        w_i = \\begin{cases}1 & |r_i| \\le \\delta \\\\ \\delta / |r_i| & |r_i| > \\delta\\end{cases}

    This gives a smooth transition between L2 (small residuals) and L1
    (large residuals) behaviour, making the solver robust to outliers while
    maintaining quadratic convergence near the optimum.

    Args:
        delta: Threshold that separates the quadratic and linear regimes.
            Residuals with ``|r| <= delta`` receive weight 1; larger residuals
            are down-weighted proportionally.  Default is ``1.0``.
        eps: Small positive constant added to the denominator for numerical
            stability.  Default is ``1e-10``.

    Returns:
        A callable ``weight_fn(residual) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residual: jax.Array) -> jax.Array:
        abs_r = jnp.abs(residual)
        return jnp.where(abs_r <= delta, jnp.ones_like(abs_r), delta / (abs_r + eps))

    return weight_fn


def irls_cauchy(c: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """Return a Cauchy (Lorentzian) IRLS weight function.

    Produces weights corresponding to the Cauchy M-estimator loss
    ``\\rho(r) = c^2 / 2 * log(1 + (r/c)^2)``:

    .. math::

        w_i = \\frac{1}{1 + (r_i / c)^2}

    The Cauchy estimator is more aggressive than Huber at down-weighting
    large residuals (sub-linear growth), giving stronger outlier rejection
    but potentially slower convergence.

    Args:
        c: Scale parameter.  Residuals much larger than ``c`` are strongly
            down-weighted.  Default is ``1.0``.

    Returns:
        A callable ``weight_fn(residual) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residual: jax.Array) -> jax.Array:
        return 1.0 / (1.0 + (residual / c) ** 2)

    return weight_fn


def irls_tukey(c: float = 4.685) -> Callable[[jax.Array], jax.Array]:
    """Return a Tukey bisquare IRLS weight function.

    Produces weights corresponding to the Tukey bisquare M-estimator:

    .. math::

        w_i = \\begin{cases}(1 - (r_i/c)^2)^2 & |r_i| \\le c \\\\ 0 & |r_i| > c\\end{cases}

    Residuals beyond the threshold ``c`` receive *zero* weight and are
    completely ignored.  This gives the strongest outlier rejection of the
    built-in estimators, but can cause instability if the initial estimate is
    poor (convergence to the correct solution is not guaranteed).

    Args:
        c: Threshold beyond which residuals are ignored.  The default value
            of ``4.685`` gives 95 % efficiency under Gaussian noise.

    Returns:
        A callable ``weight_fn(residual) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residual: jax.Array) -> jax.Array:
        u = residual / c
        return jnp.where(jnp.abs(u) <= 1.0, (1.0 - u ** 2) ** 2, jnp.zeros_like(u))

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
        A callable ``weight_fn(residual) -> weights`` suitable for
        :attr:`~jaxls.Cost.irls_weight_fn`.
    """
    def weight_fn(residual: jax.Array) -> jax.Array:
        return 1.0 / (jnp.abs(residual) + eps)

    return weight_fn
