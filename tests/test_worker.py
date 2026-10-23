# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast.constants import STACK_CHUNK
from generalforecast.worker import _stacked


def _item(index: int) -> tuple:
    return (
        jnp.full(3, index, dtype=jnp.float32),
        jnp.int32(index),
        np.full(2, index, dtype=np.float32),
    )


def test_a_stack_past_a_chunk_matches_a_stack_of_every_item() -> None:
    items = [_item(index) for index in range(2 * STACK_CHUNK + 5)]
    stacked = _stacked(items)
    for leaf, column in zip(stacked, zip(*items, strict=True), strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.stack(column))
        assert leaf.shape == (len(items), *np.shape(column[0]))


def test_a_new_count_past_a_chunk_compiles_only_its_join() -> None:
    _stacked([_item(index) for index in range(STACK_CHUNK)])
    compiled: list[str] = []

    def compile_event(event: str, duration_secs: float, **details: Any) -> None:
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(details.get("fun_name", ""))

    jax.config.update("jax_enable_compilation_cache", False)
    jax.monitoring.register_event_duration_secs_listener(compile_event)
    try:
        _stacked([_item(index) for index in range(16 * STACK_CHUNK)])
    finally:
        jax.monitoring.unregister_event_duration_listener(compile_event)
        jax.config.update("jax_enable_compilation_cache", True)
    assert compiled == ["jit(_join_columns)"]
