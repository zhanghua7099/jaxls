import contextlib
import inspect
import time
from functools import partial
from typing import Any

import jax
import numpy as onp
import termcolor
from jax import numpy as jnp
from loguru import logger


def tikhonov_floor(dtype: Any) -> Any:
    eps = float(onp.finfo(dtype).eps)
    return eps * (2e4 if onp.dtype(dtype) == onp.float32 else 4.0)


def _batched_gram(a: Any, b: Any) -> Any:
    return jnp.sum(a[..., :, :, None] * b[..., :, None, :], axis=-3)


def _batched_outer_last(a: Any, b: Any) -> Any:
    return jnp.sum(a[..., :, None, :] * b[..., None, :, :], axis=-1)


def _batched_matmul(a: Any, b: Any) -> Any:
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


@contextlib.contextmanager
def stopwatch(label: Any = "unlabeled block") -> Any:
    start_time = time.time()
    print("\n========")
    print(f"Running ({label})")
    yield
    print(f"{termcolor.colored(str(time.time() - start_time), attrs=['bold'])} seconds")
    print("========")


def _log(fmt: Any, *args, **kwargs) -> Any:
    logger.bind(function="log").info(fmt, *args, **kwargs)


def jax_log(fmt: Any, *args, **kwargs) -> Any:
    jax.debug.callback(partial(_log, fmt), *args, **kwargs)


def print_deprecation_warning(
    message0: Any, message1: Any = None, stack_level: Any = 2
) -> Any:

    frame = inspect.currentframe()
    for _ in range(stack_level):
        if not frame:
            return
        frame = frame.f_back

    if frame is None:
        return

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

    if frame_info.code_context:
        code_lines = frame_info.code_context
        start_line = lineno - (len(code_lines) // 2)

        while start_line < 1:
            start_line += 1
            code_lines.pop()

        code = "".join(code_lines)

        highlight_line = start_line + (lineno - start_line)

        syntax = Syntax(
            code,
            "python",
            line_numbers=True,
            start_line=start_line,
            highlight_lines={highlight_line},
        )
        panel_content.append(syntax)

    if message1 is not None:
        panel_content.append(Text.from_markup(message1))

    content_group = Group(*panel_content)

    console.print(
        Panel(
            content_group,
            title="[bold red]Deprecation Warning[/bold red]",
            border_style="red",
            expand=False,
        )
    )


_MAD_CONSISTENCY_FACTOR = 0.6745


def irls_huber(delta: Any = 1.0, eps: Any = 1e-10) -> Any:
    def weight_fn(residuals: Any) -> Any:
        def _per_instance(r: Any) -> Any:
            abs_r = jnp.abs(r)
            return jnp.where(
                abs_r <= delta, jnp.ones_like(abs_r), delta / (abs_r + eps)
            )

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_cauchy(c: Any = 1.0) -> Any:
    def weight_fn(residuals: Any) -> Any:
        def _per_instance(r: Any) -> Any:
            return 1.0 / (1.0 + (r / c) ** 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_tukey(c: Any = 4.685) -> Any:
    def weight_fn(residuals: Any) -> Any:
        def _per_instance(r: Any) -> Any:
            u = r / c
            return jnp.where(jnp.abs(u) <= 1.0, (1.0 - u**2) ** 2, jnp.zeros_like(u))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_l1(eps: Any = 1e-6) -> Any:
    def weight_fn(residuals: Any) -> Any:
        def _per_instance(r: Any) -> Any:
            return 1.0 / (jnp.abs(r) + eps)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_huber_adaptive(k: Any = 1.345, eps: Any = 1e-10) -> Any:
    def weight_fn(residuals: Any) -> Any:

        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: Any) -> Any:
            u = jnp.abs(r) / sigma
            return jnp.where(u <= k, jnp.ones_like(u), k / (u + eps))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_cauchy_adaptive(k: Any = 2.385, eps: Any = 1e-10) -> Any:
    def weight_fn(residuals: Any) -> Any:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: Any) -> Any:
            u = r / sigma
            return 1.0 / (1.0 + (u / k) ** 2)

        return jax.vmap(_per_instance)(residuals)

    return weight_fn


def irls_tukey_adaptive(k: Any = 4.685, eps: Any = 1e-10) -> Any:
    def weight_fn(residuals: Any) -> Any:
        sigma = jnp.median(jnp.abs(residuals.flatten())) / _MAD_CONSISTENCY_FACTOR
        sigma = jnp.maximum(sigma, eps)

        def _per_instance(r: Any) -> Any:
            u = r / sigma
            t = u / k
            return jnp.where(jnp.abs(t) <= 1.0, (1.0 - t**2) ** 2, jnp.zeros_like(t))

        return jax.vmap(_per_instance)(residuals)

    return weight_fn
