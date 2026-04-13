from typing import Any
import contextlib
import inspect
import time
from functools import partial

import jax
import termcolor
from jax import numpy as jnp
from loguru import logger


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


def irls_huber(delta: Any = 1.0, eps: Any = 1e-10) -> Any:
    def weight_fn(residual: Any) -> Any:
        abs_r = jnp.abs(residual)
        return jnp.where(abs_r <= delta, jnp.ones_like(abs_r), delta / (abs_r + eps))

    return weight_fn


def irls_cauchy(c: Any = 1.0) -> Any:
    def weight_fn(residual: Any) -> Any:
        return 1.0 / (1.0 + (residual / c) ** 2)

    return weight_fn


def irls_tukey(c: Any = 4.685) -> Any:
    def weight_fn(residual: Any) -> Any:
        u = residual / c
        return jnp.where(jnp.abs(u) <= 1.0, (1.0 - u**2) ** 2, jnp.zeros_like(u))

    return weight_fn


def irls_l1(eps: Any = 1e-6) -> Any:
    def weight_fn(residual: Any) -> Any:
        return 1.0 / (jnp.abs(residual) + eps)

    return weight_fn
