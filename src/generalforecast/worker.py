# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The worker: runs the steps a node schedules, on one accelerator.

The node decides what to compute and where state lives. The worker holds
everything with data in it: each table's fitted state, and pages of row
representations and encoded contexts packed into the slots the node assigns,
in one pool of device memory and one of host memory, each as large as the
node's tier. Work from many tables that shares a model, task and padded shapes
runs as one vectorized call.
"""

import contextlib
import dataclasses
import functools
import hashlib
import logging
import os
import platform
import re
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as finished
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from generalforecast import _core, protocol
from generalforecast._core import CacheKey, bucket_size
from generalforecast.constants import (
    AUTOTUNE_SHARE,
    FIT_CELL_BYTES,
    FIT_ROW_SCALE,
    FUSED_STACKS,
    MEASURED_ROWS,
    SCATTER_ROWS,
    SCATTER_THREADS,
    STACK_CHUNK,
    TABLE_STAGES,
    WARM_RUN_ROWS,
    WARMED_PAGES,
    WARMED_QUERIES,
)
from generalforecast.models.contract import TabularFunctions, TabularProperties
from generalforecast.models.series import SeriesFunctions, SeriesGroup, SeriesProperties
from generalforecast.runtime import shipped
from generalforecast.runtime.budget import (
    Call,
    Chunks,
    MemoryBudget,
    _shapes,
    out_of_memory,
)
from generalforecast.runtime.decoding import (
    PROGRAMS,
    Context,
    Decoding,
    Group,
    alike,
    decoding_of,
    merged,
    packed,
    paged_decode,
    paged_inputs,
    paged_shape,
    query_rows,
    split,
)
from generalforecast.runtime.device import (
    DeviceSettings,
    cache_root,
    configure,
    cpu_flags,
    load_cache,
    portable_cpu_programs,
    store_cache,
    to_cache_dtype,
    use_compile_cache,
    with_summary_chunk,
    without_command_buffers,
)
from generalforecast.runtime.memory import (
    EncodeProfile,
    FitProfile,
    MemoryProfile,
    NoMemoryAnalysis,
    WorkingProfile,
    decode_profile,
    encode_bytes,
    encode_profile,
    encode_widths,
    fit_profile,
    memory_profile,
    tree_bytes,
)
from generalforecast.runtime.paging import (
    CacheLayout,
    DevicePool,
    DiskPool,
    HostPool,
    PoolLost,
    StateLayout,
    cache_layout,
    pack,
    pack_state,
    state_layout,
    unpack_states,
)
from generalforecast.runtime.programs import (
    Kind,
    Programs,
    abstract,
    counts_up_to,
    ladder,
    most_rows,
    zeros,
)
from generalforecast.runtime.programs import (
    signature as kind_signature,
)
from generalforecast.runtime.series import (
    SeriesBytes,
    abstract_group,
    buckets,
    fixed_leaves,
    horizon_bucket,
    laid_group,
    patch_bucket,
    series_bytes,
    series_decode,
    series_extend,
    series_layout,
)
from generalforecast.runtime.shapes import padded as padded_items
from generalforecast.system import usable_cpus

logger = logging.getLogger(__name__)

# A step comes from the bindings in-process, or decoded from a server's frame.
Step = _core.Step | protocol.Step
Fit = _core.Fit | protocol.Fit
Page = _core.Page | protocol.Page
Unit = _core.Unit | protocol.Unit
QueryBatch = _core.QueryBatch | protocol.QueryBatch
Movement = _core.Movement | protocol.Movement
SeriesUnit = _core.SeriesUnit | protocol.SeriesUnit
Forecast = _core.Forecast | protocol.Forecast


@dataclass
class _Deferred:
    """A unit there is no room to keep, encoded only when its decode needs it.

    Attributes:
        table: The unit's table.
        arguments: Its encode's arguments after the model parameters.
        rows: Its real rows.
        stage: `encode`, or `fit_encode` for a unit fit on its own rows.
        encoded: Whether it was encoded once already, for the counts.
    """

    table: int
    arguments: tuple
    rows: int
    stage: str = "encode"
    encoded: bool = False


# Units without slots, put off until decoded.
Transient = dict[int, _Deferred]
# A page's row representations, on the device or the host, and its targets.
Representations = tuple[Any, np.ndarray]


@dataclass
class WorkerStats:
    """Counters across every step.

    Attributes:
        encoded: Units encoded, whether kept or used once.
        reused: Units read from the cache instead of encoded, an extended
            unit's prefix among them.
        extended: Series units grown from a cached encoding by new patches.
        transient: Encoded units there was no room to keep.
        promoted: Cached units moved back from host to device memory.
        demoted: Cached units moved from device to host memory.
        spilled: Cached contexts moved from host memory to disk.
        dropped: Cached contexts dropped, evicted from the host or from a past epoch.
        rows_encoded: Context rows encoded across all units.
        rows_embedded: Rows embedded outside a fit, for representation pages
            that were not cached. Rows a page copies from its earlier entry
            count under rows_copied, not here.
        rows_copied: Page rows copied from the page's earlier entry instead
            of embedded, after other rows of the page were written.
        rows_reused: Context rows in units read from the cache.
        steps: Steps computed.
        queries: Query rows decoded.
        calls: Compiled calls per stage; tables batched together share one.
        compiles: Calls per stage whose shape their program had not seen,
            which compiled, or came from the compile cache.
        seconds: Time in compiled calls per stage, compiling included, since
            each call waits for its results.
        parts: Times each part of a step ran, by name. The parts split a
            step's time without overlap: `frame` decodes a server's step
            frame and `reply` converts predictions into its reply, `move`
            applies the step's copies and drops, `pad` pads rows to their
            bucket on the host, `stack` stacks a call's items, `compile`
            traces and compiles a call whose shape its program had not seen
            and hands it to the device, `dispatch` hands a call its program
            has seen and its host arrays to the device, `wait` waits for the
            device to finish it, `unstack` takes each item's outputs from
            the call's, `fetch` copies results to the host, `store` packs
            contexts and pages into their slots, and `other` is the rest of
            the step, the Python between the parts and XLA's measures of a
            call's memory. A step that fails keeps no part but `other`,
            which takes its whole time. The compiled stages' `seconds`
            overlap the parts: a call's seconds are its `compile` or
            `dispatch`, its `wait` and its measure.
        part_seconds: Time in each part.
        timers: Whether parts are timed. Untimed, `part` and `timed` count
            nothing and read no clock.
    """

    encoded: int = 0
    reused: int = 0
    extended: int = 0
    transient: int = 0
    promoted: int = 0
    demoted: int = 0
    spilled: int = 0
    dropped: int = 0
    rows_encoded: int = 0
    rows_embedded: int = 0
    rows_copied: int = 0
    rows_reused: int = 0
    steps: int = 0
    queries: int = 0
    calls: dict[str, int] = field(default_factory=dict)
    compiles: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    parts: dict[str, int] = field(default_factory=dict)
    part_seconds: dict[str, float] = field(default_factory=dict)
    timers: bool = False

    def part(self, name: str, seconds: float) -> None:
        """Counts one run of a part of a step, and the seconds it took."""
        if not self.timers:
            return
        with _COUNTING:
            self.parts[name] = self.parts.get(name, 0) + 1
            self.part_seconds[name] = self.part_seconds.get(name, 0.0) + seconds

    def timed(self, name: str) -> contextlib.AbstractContextManager[None]:
        """Counts the block's time as one run of a part of a step."""
        return self._timing(name) if self.timers else _UNTIMED

    @contextlib.contextmanager
    def _timing(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.part(name, time.perf_counter() - started)

    def snapshot(self) -> "WorkerStats":
        """A copy to count from, or to go back to."""
        return dataclasses.replace(
            self,
            calls=dict(self.calls),
            compiles=dict(self.compiles),
            seconds=dict(self.seconds),
            parts=dict(self.parts),
            part_seconds=dict(self.part_seconds),
        )

    def stages_since(self, before: "WorkerStats") -> dict[str, tuple[int, float]]:
        """Each stage's compiled calls and their seconds since `before`."""
        return _since(before.calls, before.seconds, self.calls, self.seconds)

    def parts_since(self, before: "WorkerStats") -> dict[str, tuple[int, float]]:
        """Each part's runs and their seconds since `before`."""
        return _since(before.parts, before.part_seconds, self.parts, self.part_seconds)


# Guards the counts of calls that run side by side; see `Worker._alone`.
_COUNTING = threading.Lock()

# What `WorkerStats.timed` gives untimed: a block that counts nothing.
_UNTIMED = contextlib.nullcontext()


def _since(
    counts_before: dict[str, int],
    seconds_before: dict[str, float],
    counts: dict[str, int],
    seconds: dict[str, float],
) -> dict[str, tuple[int, float]]:
    """Each name's count and seconds that rose between two readings."""
    return {
        name: (
            count - counts_before.get(name, 0),
            seconds[name] - seconds_before.get(name, 0.0),
        )
        for name, count in counts.items()
        if count > counts_before.get(name, 0)
    }


@dataclass
class _Model[Functions: (TabularFunctions, SeriesFunctions)]:
    functions: Functions
    parameters: Any


_TabularModel = _Model[TabularFunctions]
_SeriesModel = _Model[SeriesFunctions]


@dataclass
class _Table:
    model: int
    num_classes: int
    features: int
    # Which columns are categorical, or None if none are.
    categorical: np.ndarray | None = None
    # Rows every page embed pads to at least, so pages that split or shrink
    # keep one compiled embed.
    page_size: int = 1
    state: Any = None
    # How each unit's own state packs into slots, for a table that summarizes
    # each context; None for one that summarizes the table.
    own: StateLayout | None = None
    # Whether the table holds series, forecast by a forecaster.
    series: bool = False
    # Whether a fit already logged that the state keeps fewer categories than
    # a column holds.
    crowded: bool = False


@dataclass
class _Page:
    """Where a page's representations are in host slots, and its targets."""

    slots: list[int]
    shape: tuple[int, ...]
    dtype: np.dtype
    targets: np.ndarray


@dataclass(frozen=True)
class _Dense:
    """A batch's queries that read a context there was no room to keep.

    Attributes:
        batch: The batch.
        rows: Its embedded query rows.
        unit: The unit the queries read.
        queries: The queries' positions in the batch.
    """

    batch: QueryBatch
    rows: Any
    unit: int
    queries: list[int]

    def split(self, limit: int | None) -> list["_Dense"]:
        """The queries in parts of at most `limit`, or whole without one."""
        if limit is None or len(self.queries) <= limit:
            return [self]
        return [
            dataclasses.replace(self, queries=self.queries[start : start + limit])
            for start in range(0, len(self.queries), limit)
        ]


def _signature(arguments: tuple) -> tuple:
    """What a call compiles to: the arguments' structure, shapes and dtypes.

    Read from the leaves' metadata, so no device array is copied to the host.
    """
    leaves, structure = jax.tree.flatten(arguments)
    return structure, tuple(
        (np.shape(leaf), np.dtype(getattr(leaf, "dtype", type(leaf))).str)
        for leaf in leaves
    )


class Worker:
    """Runs steps for the tables a node assigns it."""

    def __init__(
        self,
        settings: DeviceSettings | None = None,
        bucket_minimum: int = 8,
        slot_bytes: int = 1 << 20,
        disk_path: Path | None = None,
        compile_cache: bool = True,
        budget: MemoryBudget | None = None,
        compile_ahead: bool = True,
        timers: bool = False,
    ) -> None:
        """Sets up an empty worker.

        Args:
            settings: Device settings for precision and attention, if any.
            bucket_minimum: Smallest padded size; must match the node's tables.
            slot_bytes: Bytes in a cache slot; must match the node's.
            disk_path: A file for the disk tier, if the node has one.
            compile_cache: Keep compiled programs on disk for later processes,
                through JAX's process-wide cache; see `use_compile_cache`.
            budget: How the worker divides its memory; this machine's if None.
            compile_ahead: Compile sizes a table may need next in quiet gaps
                (see `Programs`); without it, each compiles when first called.
            timers: Time each part of a step; see `WorkerStats.parts`.
        """
        without_command_buffers()
        portable_cpu_programs()
        self._settings, self._minimum = settings, bucket_minimum
        self._slot_bytes = slot_bytes
        # On a CPU a stacked fused call takes as long per unit as one alone,
        # and pads to its bucket and compiles for it, so units run one a
        # call, side by side on the worker's cores: two to four at once
        # served 20 to 40% more than one at a time on four cores.
        self._stacks = jax.default_backend() != "cpu"
        self._side_by_side = 1 if self._stacks else usable_cpus()
        self._alongside: ThreadPoolExecutor | None = None
        # NumPy copies a fancy index's rows without the GIL, so answers
        # scatter on several cores.
        self._scatter_threads = min(SCATTER_THREADS, usable_cpus())
        self._scatterers: ThreadPoolExecutor | None = None
        # Takes the keys of programs readied beside steps for deferred
        # tables; without it, no table's part waits for a program.
        self.announce: Callable[[list[str]], None] | None = None
        # The tables the running step defers, by the program they wait for.
        self.deferred: dict[int, str] = {}
        # Programs being readied beside steps, and those that failed to be,
        # which the steps that need them compile and report.
        self._preparing: set[str] = set()
        self._unpreparable: set[str] = set()
        # On an accelerator a program compiles for tens of seconds, so a
        # stacked count not yet compiled compiles on the side; see `Chunks.run`.
        self._on_side = self._stacks and compile_ahead
        if compile_cache:
            use_compile_cache()
        # Where measured fits are kept for later processes, beside compiled programs.
        self._fit_cache = cache_root() / "fit" if compile_cache else None
        # How the worker divides its memory; calls are sized from it as they run.
        self.budget = budget or MemoryBudget.measure()
        self._chunks = Chunks(
            self._count, self._compiled, self._part, self._seen, self._noted
        )
        self._manifest_dir = (
            _manifests(bucket_minimum, slot_bytes) if compile_cache else None
        )
        # The worker's compiled programs, and those it compiles ahead.
        self.programs = Programs(
            settings,
            lambda: self.budget.short_of_memory(),
            (
                _item,
                _only_item,
                _stack_columns,
                store_cache,
                pack,
                pack_state,
                unpack_states,
                *PROGRAMS,
            ),
            ahead=compile_ahead,
            manifests=self._manifest_dir,
        )
        self._models: dict[int, _Model] = {}
        # The spec each model was loaded from, under which its kinds of call are recorded.
        self._specs: dict[int, str] = {}
        self._tables: dict[int, _Table] = {}
        self._layouts: dict[tuple[int, int], CacheLayout] = {}
        self._profiles: dict[tuple[int, int], MemoryProfile] = {}
        self._decodes: dict[int, WorkingProfile] = {}
        self._encodes: dict[
            tuple[int, bool, tuple[int, int]], EncodeProfile | None
        ] = {}
        self._encode_bytes: dict[tuple[int, bool, int, int], int | None] = {}
        self._fit_profiles: dict[tuple[int, int], FitProfile] = {}
        self._checked: set[tuple[int, int, int]] = set()
        self._disk_path = disk_path
        self._slots = (0, 0, 0)
        self.allocate(0, 0, 0)
        self._contexts: dict[CacheKey, Context] = {}
        # The leaves a series cache keeps beside its patches, by entry.
        self._fixed: dict[CacheKey, list[Any]] = {}
        self._series_layouts: dict[tuple[int, int], CacheLayout] = {}
        self._series_profiles: dict[int, SeriesBytes] = {}
        self._pages: dict[CacheKey, _Page] = {}
        self.stats = WorkerStats()
        self.timers = timers

    @property
    def timers(self) -> bool:
        """Whether each part of a step is timed; see `WorkerStats.parts`."""
        return self.stats.timers

    @timers.setter
    def timers(self, timers: bool) -> None:
        self.stats.timers = timers
        self._chunks.timed = self._part if timers else None

    def allocate(self, device_slots: int, host_slots: int, disk_slots: int) -> None:
        """Sizes the pools as the node's tiers; call it before the first step.

        Args:
            device_slots: Slots in the node's device tier.
            host_slots: Slots in its host tier.
            disk_slots: Slots in its disk tier, if the worker has a disk path.
        """
        self._slots = (device_slots, host_slots, disk_slots)
        self._pool_bytes = device_slots * self._slot_bytes
        # What calls that ran out of memory beside the old pool taught no longer holds.
        self._chunks.relax()
        # The old pool goes first, so the device never holds two at once.
        if hasattr(self, "_device"):
            self._device.release()
        self._device = DevicePool(self._slot_bytes, device_slots)
        self._host = HostPool(self._slot_bytes, host_slots)
        self._disk = (
            None
            if self._disk_path is None
            else DiskPool(self._slot_bytes, disk_slots, self._disk_path)
        )

    @property
    def allocated(self) -> bool:
        """Whether `allocate` sized the pools, which steps and `warm_start` read."""
        return hasattr(self, "_device")

    @property
    def _call_bytes(self) -> int:
        """Bytes one compiled call may take beside the device pool."""
        return self.budget.call_bytes(self._pool_bytes)

    @property
    def _window_bytes(self) -> int:
        """Bytes of contexts a decode gathers before its call."""
        return self.budget.window_bytes(self._pool_bytes)

    @property
    def lost(self) -> bool:
        """Whether the device pool lost its contexts; see `forget_cache`."""
        return self._device.lost

    def close(self) -> None:
        """Frees the device pool and forgets every cached context and page."""
        if self._alongside is not None:
            self._alongside.shutdown()
            self._alongside = None
        if self._scatterers is not None:
            self._scatterers.shutdown()
            self._scatterers = None
        self._contexts.clear()
        self._fixed.clear()
        self._pages.clear()
        self._device.release()

    def forget_cache(self) -> None:
        """Drops every cached context and page, with fresh pools.

        The node's `forget_cache` does the same on its side.
        """
        self._contexts.clear()
        self._fixed.clear()
        self._pages.clear()
        self.allocate(*self._slots)

    def add_model(
        self,
        model: int,
        functions: TabularFunctions | SeriesFunctions,
        parameters: Any,
        spec: str | None = None,
    ) -> None:
        """Loads a model's parameters under the node's model id.

        With `spec`, what the model was loaded from, the kinds of call it
        compiles for are recorded under it, and `warm_start` compiles them.
        """
        if self._settings is not None:
            parameters = configure(parameters, self._settings)
        self._models[model] = _Model[Any](functions, parameters)
        if spec is not None:
            self._specs[model] = spec
            manifest = self.programs.manifest(spec)
            if manifest is not None and not manifest.path.exists():
                # A machine that never ran the model starts from what its
                # release compiled on this platform, if it shipped any.
                assert self._manifest_dir is not None
                shipped.install(cache_root(), self._manifest_dir.name, spec)

    def _measured(self, model: int, key: tuple, measure: Callable[[], Any]) -> Any:
        """What `measure` returns, kept for later processes under the model's spec and settings."""
        spec = self._specs.get(model)
        if spec is None:
            return measure()
        return self.programs.measured(spec, (*key, repr(self._settings)), measure)

    def warm_start(
        self,
        progress: Callable[[], None] | None = None,
        models: Sequence[int] | None = None,
    ) -> None:
        """Compiles the kinds of call earlier processes recorded for the loaded models, before any step.

        Each kind compiles for every power of two of items up to the most a
        call of it took, and for each count it took, from shapes alone, so
        the device stays free. Programs JAX's compile cache holds load from
        it; tracing them is what remains, which steps would otherwise do.
        A kind that no longer compiles, as after the model's code changed,
        is skipped. Call it after `allocate`: paged decodes read the pool.

        Args:
            progress: Called after each program compiles.
            models: The loaded models to compile for; every one by default.
        """
        started = time.perf_counter()
        chosen = self._specs.keys() if models is None else models
        for model, spec in [(model, self._specs[model]) for model in chosen]:
            parameters = self._models[model].parameters
            recorded = self.programs.recorded(spec)
            if not self.forecaster(model) and recorded.kinds:
                # What a step measures by compiling, once per model and width,
                # before it decides where a context goes.
                self.decode_profile(model)
                for measure in recorded.measures:
                    self._measure(model, measure)
                    if progress is not None:
                        progress()
            for kind, counts in recorded.kinds.items():
                program = (kind.program[0], model, *kind.program[2:])
                try:
                    function = cast(jax.stages.Wrapped, self._program(program))
                    stacks = self._buckets(kind.program[0])
                    for count in stacks or counts_up_to(counts):
                        items = abstract(kind.item, count)
                        shared = self._shared(program, kind, items)
                        function.lower(parameters, *shared, *items).compile()
                        # A program's first run sets it up on the device, which
                        # would otherwise slow the first step that calls it. One
                        # reading a large table takes as long on zeros as the
                        # call it would speed up, so only compiles.
                        inputs = zeros((shared, items))
                        if inputs is not None and (
                            kind.program[0] not in TABLE_STAGES
                            or most_rows(kind.item) <= WARM_RUN_ROWS
                        ):
                            jax.block_until_ready(
                                function(parameters, *inputs[0], *inputs[1])
                            )
                        self.programs.warmed += 1
                        if progress is not None:
                            progress()
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        "not compiling a recorded %s call: %s", kind.program[0], error
                    )
        self.programs.warm_seconds += time.perf_counter() - started

    def _remember(self, model: int, measure: tuple) -> None:
        """Records a measure a model took by compiling, for `warm_start` to take again."""
        if model in self._specs:
            self.programs.remember_measure(self._specs[model], measure)

    def _measure(self, model: int, measure: tuple) -> None:
        """Takes a recorded measure again: a width's sizes and layout, or an encode's bytes."""
        try:
            match measure:
                case ("width", int(features)):
                    self.profile(model, features)
                    self.layout(model, features)
                case ("encode", int(features), int(rows), bool(fit)):
                    self.encode_bytes(model, features, rows, fit)
        except Exception as error:  # noqa: BLE001
            logger.warning("not measuring a recorded %s: %s", measure[0], error)

    def _shared(self, program: tuple, kind: Kind, items: tuple) -> tuple:
        """A recorded kind's shared arguments for `items`: a paged decode reads this pool."""
        if program[0] != "paged":
            return abstract(kind.shared)
        blocks = items[2]
        pool = self._device.data
        if isinstance(pool, np.ndarray):
            # A CPU's decode reads a copy of its slots; see `DevicePool.source`.
            rows = bucket_size(int(np.prod(blocks.shape)), 1)
            source = jax.ShapeDtypeStruct((rows, pool.shape[1]), np.uint8)
        else:
            source = jax.ShapeDtypeStruct(pool.shape, pool.dtype)
        return (source,)

    def _seen(self, call: Call) -> None:
        """Records a call that compiled, if it is a kind a later process can compile again."""
        if call.program is None or call.program[1] not in self._specs:
            return
        # What a paged decode reads its slots from follows from its items and
        # the pool; see `_shared`.
        shared = () if call.program[0] == "paged" else call.shared[1:]
        kind = Kind(
            (call.program[0], None, *call.program[2:]),
            kind_signature(shared),
            kind_signature(call.items, stacked=True),
        )
        self.programs.remember(self._specs[call.program[1]], kind, call.padded)

    def _noted(self, stage: str, key: Hashable, call: Call, seconds: float) -> None:
        """Logs a call that compiled, with its shapes, while parts are timed."""
        if self.stats.timers:
            logger.warning(
                "compiled %s %s at %d items in %.2f s: items %s, kind %s",
                stage,
                call.program,
                call.padded,
                seconds,
                kind_signature(call.items, stacked=True),
                key,
            )

    def forecaster(self, model: int) -> bool:
        """Whether a loaded model is a forecaster."""
        return isinstance(self._models[model].functions, SeriesFunctions)

    def properties(self, model: int) -> TabularProperties:
        """A loaded tabular model's properties."""
        entry = self._tabular(model)
        return entry.functions.properties(entry.parameters)

    def series_extendable(self, model: int) -> bool:
        """Whether a forecaster grows an encoding by whole patches of new steps (`extend`)."""
        return self._series(model).functions.extend is not None

    def series_properties(self, model: int) -> SeriesProperties:
        """A loaded forecaster's properties."""
        entry = self._series(model)
        return entry.functions.properties(entry.parameters)

    def _tabular(self, model: int) -> _TabularModel:
        """A loaded tabular model."""
        entry = self._models[model]
        if isinstance(entry.functions, SeriesFunctions):
            raise ValueError(  # noqa: TRY004
                f"model {model} is a forecaster, which only a series table uses"
            )
        return cast(_TabularModel, entry)

    def _series(self, model: int) -> _SeriesModel:
        """A loaded forecaster."""
        entry = self._models[model]
        if not isinstance(entry.functions, SeriesFunctions):
            raise ValueError(  # noqa: TRY004
                f"model {model} is not a forecaster, which a series table needs"
            )
        return cast(_SeriesModel, entry)

    def profile(self, model: int, features: int) -> MemoryProfile:
        """A loaded model's sizes for tables with `features` columns, as stored."""
        if (model, features) not in self._profiles:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._profiles[model, features] = self._measured(
                model,
                ("profile", features),
                lambda: memory_profile(
                    entry.functions, entry.parameters, features, cache_dtype
                ),
            )
        return self._profiles[model, features]

    def series_profile(self, model: int) -> SeriesBytes:
        """A loaded forecaster's cache bytes per row and patch, as stored."""
        if model not in self._series_profiles:
            entry = self._series(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._series_profiles[model] = series_bytes(
                entry.functions, entry.parameters, cache_dtype
            )
        return self._series_profiles[model]

    def _fits_device_series(self, model: int, rows: int, steps: int) -> None:
        """Refuses a group whose cache alone would not fit the device's memory.

        Raises:
            ValueError: Naming what reads less.
        """
        sizes = self.series_profile(model)
        patches = steps // self.series_properties(model).patch
        bytes = patches * (sizes.per_row * rows + sizes.per_patch)
        bytes += sizes.fixed_per_row * rows + sizes.fixed
        if bytes > self.budget.device:
            raise ValueError(
                f"a group of {rows} rows over {steps} steps takes {bytes >> 20} MiB "
                f"as the model caches it, but the device holds "
                f"{self.budget.device >> 20} MiB; use smaller groups, fewer "
                "covariates, or a device with more memory"
            )

    def check_series(self, model: int, features: int, rows: int = 2) -> None:
        """Traces a forecast of a small group, without running it.

        Raises:
            ValueError: If the model refuses such a group.
        """
        entry = self._series(model)
        properties = entry.functions.properties(entry.parameters)
        group = abstract_group(rows * (1 + features), 2 * properties.patch)
        try:
            jax.eval_shape(
                lambda parameters, group: entry.functions.forecast(
                    parameters, group, properties.output_patch
                ),
                entry.parameters,
                group,
            )
        except ValueError as error:
            # A model's cap on a group's rows, worded as the planner words
            # its refusal; anything else is the model's own complaint.
            if "passes the model's" not in str(error):
                raise
            raise ValueError(
                f"a group of {rows} series takes {rows * (1 + features)} rows with "
                f"the covariates, past what the model serves per group ({error})"
            ) from error

    def _series_layout(self, model: int, rows: int) -> CacheLayout:
        """How a forecaster's caches pack into slots, for groups of `rows` padded rows."""
        if (model, rows) not in self._series_layouts:
            entry = self._series(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._series_layouts[model, rows] = series_layout(
                entry.functions, entry.parameters, rows, self._slot_bytes, cache_dtype
            )
        return self._series_layouts[model, rows]

    def decode_profile(self, model: int) -> WorkingProfile:
        """A loaded model's decode of its smallest query chunk, measured once."""
        if model not in self._decodes:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._decodes[model] = self._measured(
                model,
                ("decode", self._size(1)),
                lambda: decode_profile(
                    entry.functions,
                    entry.parameters,
                    self._stage("decode", model, 0),
                    self._size(1),
                    cache_dtype,
                ),
            )
        return self._decodes[model]

    def encode_profile(
        self, model: int, features: int, fit: bool = False
    ) -> EncodeProfile | None:
        """A loaded model's encode of a context, measured once for widths around `features`.

        With `fit`, the encode fits the context's own state first. None where
        XLA cannot measure it.
        """
        widths = encode_widths(features)
        if (model, fit, widths) not in self._encodes:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype

            def measure() -> EncodeProfile | None:
                try:
                    return encode_profile(
                        entry.functions, entry.parameters, fit, cache_dtype, widths
                    )
                except Exception as error:  # noqa: BLE001
                    logger.warning(
                        "could not measure the model's encode at widths %s (%s); "
                        "each context's is measured at its own size",
                        widths,
                        error,
                    )
                    return None

            self._encodes[model, fit, widths] = self._measured(
                model, ("encode_profile", fit, widths), measure
            )
        return self._encodes[model, fit, widths]

    def encode_bytes(
        self, model: int, features: int, rows: int, fit: bool = False
    ) -> int | None:
        """Bytes a loaded model's encode of a context of `rows` padded rows takes at most.

        From the model's profile where it holds, and otherwise measured at the
        context's own width and size. None where XLA gives no measure.

        Raises:
            ValueError: If the encode cannot even be compiled to measure it.
        """
        per_row = self.profile(model, features).cache_bytes_per_row
        if rows >= MEASURED_ROWS:
            profile = self.encode_profile(model, features, fit)
            if profile is not None:
                return profile.working_bytes(per_row, rows)
        key = (model, fit, features, rows)
        if key not in self._encode_bytes:
            self._remember(model, ("encode", features, rows, fit))
            entry = self._tabular(model)
            try:
                cache_dtype = (
                    None if self._settings is None else self._settings.cache_dtype
                )
                measured = self._measured(
                    model,
                    ("encode", features, rows, fit),
                    lambda: encode_bytes(
                        entry.functions,
                        entry.parameters,
                        features,
                        rows,
                        fit,
                        cache_dtype,
                    ),
                )
            except NoMemoryAnalysis as error:
                logger.warning(
                    "could not measure the model's encode (%s); contexts are "
                    "refused only by their decode",
                    error,
                )
                measured = None
            except Exception as error:
                raise ValueError(
                    f"could not compile the encode of a context of {rows} rows to "
                    f"measure it ({error}); use a strategy that reads fewer rows, "
                    "such as neighbors, or a device with more memory"
                ) from error
            self._encode_bytes[key] = measured
        return self._encode_bytes[key]

    def fit_profile(self, model: int, features: int) -> FitProfile:
        """What a loaded model's fit takes for tables with `features` columns, measured once."""
        if (model, features) not in self._fit_profiles:
            entry = self._tabular(model)
            self._fit_profiles[model, features] = fit_profile(
                entry.functions, entry.parameters, features, self._fit_cache
            )
        return self._fit_profiles[model, features]

    def fit_rows(self, model: int, features: int) -> tuple[int, int]:
        """The most rows a fit takes within the call budget, for tables with `features` columns.

        Each is a padded size, so a fit padded to its size stays within it.

        Args:
            model: A loaded model's id.
            features: Number of feature columns.

        Returns:
            The most rows a state alone is fit on, and the most a fit embeds
            in the call that fits its state; `sys.maxsize` for any.
        """
        profile = self.fit_profile(model, features)
        return (
            self._padded_floor(profile.state_rows(self._call_bytes)),
            self._padded_floor(profile.fused_rows(self._call_bytes)),
        )

    def fit_bound(self, model: int, features: int) -> int:
        """The most rows a table's fit takes before `fit_rows` is measured, without compiling.

        A padded size at which a fit embedding its rows takes the call budget
        even at `FIT_CELL_BYTES` a cell and `FIT_ROW_SCALE` times its rows'
        representations, far more than any built-in model's fit holds.

        Args:
            model: A loaded model's id.
            features: Number of feature columns.

        Returns:
            The rows, a padded size or none.
        """
        rows = self.profile(model, features).row_bytes
        per_row = FIT_CELL_BYTES * (features + 4) + FIT_ROW_SCALE * rows
        return self._padded_floor(self._call_bytes // per_row)

    def _padded_floor(self, rows: int) -> int:
        """The largest padded size of at most `rows` rows, 0 if none, `sys.maxsize` for any."""
        if rows >= sys.maxsize:
            return sys.maxsize
        if rows < self._minimum:
            return 0
        return self._minimum << ((rows // self._minimum).bit_length() - 1)

    def _summary_chunk(self, model: int, width: int) -> int | None:
        """Cells a fit's column summary reads a step within the call budget, if it has one."""
        if not self.properties(model).column_summary:
            return None
        return self.fit_profile(model, width).summary_chunk(self._call_bytes)

    def check(self, model: int, num_classes: int, features: int) -> None:
        """Traces a fit for a table of this task and width, without running it.

        Args:
            model: A loaded model's id.
            num_classes: Number of classes, or 0 for regression.
            features: Number of feature columns.

        Raises:
            ValueError: If the model refuses such a table, so it is refused
                when created rather than at its first fit.
        """
        if (model, num_classes, features) in self._checked:
            return
        size = self._size(1)

        def check() -> bool:
            targets = jnp.zeros(size, jnp.int32 if num_classes else jnp.float32)
            jax.eval_shape(
                self._stage("fit", model, num_classes),
                self._models[model].parameters,
                jnp.zeros((size, features), jnp.float32),
                targets,
                1,
                None,
            )
            return True

        # Tracing the model takes most of a second, so a table it accepted is
        # remembered for later processes; one it refuses is traced again.
        self._measured(model, ("check", num_classes, features, size), check)
        self._checked.add((model, num_classes, features))

    def layout(self, model: int, features: int) -> CacheLayout:
        """How a loaded model's contexts pack into slots, for tables with `features` columns.

        Raises:
            ValueError: If a slot cannot hold one of the model's context rows.
        """
        if (model, features) not in self._layouts:
            entry = self._tabular(model)
            cache_dtype = None if self._settings is None else self._settings.cache_dtype
            self._layouts[model, features] = self._measured(
                model,
                ("layout", features),
                lambda: cache_layout(
                    entry.functions,
                    entry.parameters,
                    features,
                    self._slot_bytes,
                    cache_dtype,
                ),
            )
        return self._layouts[model, features]

    def add_table(
        self,
        table: int,
        model: int,
        num_classes: int,
        features: int,
        categorical: Sequence[int] = (),
        page_size: int = 1,
        summary: str = "table",
        series: bool = False,
    ) -> None:
        """Serves a table with a loaded model.

        Args:
            table: The node's id for the table.
            model: A loaded model's id.
            num_classes: Number of classes, or 0 for regression.
            features: Number of feature columns.
            categorical: Indices of the columns that hold category codes.
            page_size: The table's page size, which page embeds pad to.
            summary: `table`, or `context` to fit each unit's state on its rows.
            series: Whether the table holds series, forecast by a forecaster.

        Raises:
            ValueError: If a column is out of range, or the model has no
                column summary to fit per context.
        """
        if series:
            self._series(model)
            self._tables[table] = _Table(model, 0, features, series=True)
            return
        self._tabular(model)
        if any(not 0 <= column < features for column in categorical):
            raise ValueError(f"categorical columns must be in 0 .. {features - 1}")
        check_summary(self.properties(model), summary)
        self.programs.add_shape((model, num_classes, features))
        self._remember(model, ("width", features))
        self.layout(model, features)
        mask = None
        if categorical:
            mask = np.zeros(features, dtype=bool)
            mask[list(categorical)] = True
        own = None
        if summary == "context":
            size = self._size(1)
            regression = num_classes == 0
            state, _ = jax.eval_shape(
                self._stage("fit", model, num_classes),
                self._models[model].parameters,
                jax.ShapeDtypeStruct((size, features), jnp.float32),
                jax.ShapeDtypeStruct((size,), jnp.float32 if regression else jnp.int32),
                1,
                mask,
            )
            own = state_layout(state, self._slot_bytes)
        self._tables[table] = _Table(
            model, num_classes, features, mask, page_size=max(1, page_size), own=own
        )

    def remove_table(self, table: int) -> None:
        """Forgets a table's state; the node drops its cache entries separately."""
        del self._tables[table]

    def run(self, step: Step) -> dict[int, np.ndarray]:
        """Runs a step: copies, fits, page embeds, encodes and decodes, in that order.

        Args:
            step: The step from the node.

        Returns:
            Predictions per request id.
        """
        with self.stats.timed("move"):
            for movement in step.movements:
                self.apply(movement)
        return self._counted(step)

    def run_apart(
        self, step: protocol.Step
    ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        """Runs a step so that one table's error fails only that table.

        The step runs whole first. If that raises, each table's part runs
        alone, so the tables that work still answer.

        Args:
            step: The step from the server.

        Returns:
            Predictions per request id, and an error message per failed table.
        """
        self.deferred = {}
        with self.stats.timed("move"):
            for movement in step.movements:
                self.apply(movement)
        try:
            return self._counted(step), {}
        except PoolLost:
            # Every table reads the pool, so no part can run alone.
            raise
        except Exception as error:
            tables = _tables(step)
            if not tables:
                raise
            if len(tables) == 1:
                return {}, {tables[0]: _message(error)}
        predictions: dict[int, np.ndarray] = {}
        failed: dict[int, str] = {}
        for table in tables:
            try:
                predictions.update(self._counted(_part(step, table)))
            except Exception as error:  # noqa: BLE001
                failed[table] = _message(error)
        return predictions, failed

    def _counted(self, step: Step) -> dict[int, np.ndarray]:
        """Computes a step, keeping its counts in `stats` only if it succeeds.

        The step's time in no other part is counted as the part `other`. A
        step that fails keeps none of its counts, and its whole time is
        counted as `other`, so the parts still add up to the time it took.
        """
        before = self.stats.snapshot()
        started = time.perf_counter() if self.timers else 0.0
        try:
            with self.programs.step():
                predictions = self._compute(step)
        except BaseException:
            self.stats = before
            if self.timers:
                self.stats.part("other", time.perf_counter() - started)
            raise
        if self.timers:
            taken = time.perf_counter() - started
            in_parts = sum(
                seconds for _, seconds in self.stats.parts_since(before).values()
            )
            self.stats.part("other", taken - in_parts)
        return predictions

    def _compute(self, step: Step) -> dict[int, np.ndarray]:
        """Fits, page embeds, encodes, decodes and forecasts a step whose copies are applied."""
        pages = self._fit(step.fits, step.pages)
        pages.update(self._embed_pages(step.pages, pages))
        transient = self._encode(step.units, step.pages, pages)
        self.stats.steps += 1
        self.stats.queries += sum(len(batch.units) for batch in step.queries)
        self.stats.queries += len(step.forecasts)
        fused = {
            index for index, unit in enumerate(step.units) if unit.action == "fuse"
        }
        reads = [batch for batch in step.queries if fused.isdisjoint(batch.units)]
        alone = [batch for batch in step.queries if not fused.isdisjoint(batch.units)]
        predictions = self._decode(reads, step.units, transient)
        predictions.update(self._decode_fused(alone, step.units))
        predictions.update(self._forecast(step.series_units, step.forecasts))
        return predictions

    def apply(self, movement: Movement) -> None:
        """Applies one copy or drop from the node.

        A drop may name an entry a failed step never stored, and a copy one
        the node dropped since it planned the copy, when the step that was to
        encode it failed meanwhile; both are skipped.
        """
        # Pages are only ever dropped.
        if movement.key.kind == "rows":
            self._pages.pop(movement.key, None)
            return
        if movement.target is None:
            self._fixed.pop(movement.key, None)
            if self._contexts.pop(movement.key, None) is not None:
                self.stats.dropped += 1
            return
        context = self._contexts.get(movement.key)
        if context is None:
            return
        slots = list(movement.slots)
        if movement.source == "device":
            data = self._device.read_host(context.slots)
        else:
            data = self._below(movement.source).read(context.slots)
        if movement.target == "device":
            self._device.write(slots, data)
            self.stats.promoted += 1
        else:
            self._below(movement.target).write(slots, data)
            if movement.target == "host":
                self.stats.demoted += 1
            else:
                self.stats.spilled += 1
        context.slots = slots

    def _below(self, tier: protocol.Tier) -> HostPool:
        """The host pool, or the disk pool."""
        if tier == "host":
            return self._host
        if self._disk is None:
            raise RuntimeError(
                "the node moved a context to disk, but the worker has no disk tier"
            )
        return self._disk

    def _size(self, rows: int) -> int:
        return bucket_size(rows, self._minimum)

    def _call(
        self,
        stage: str,
        jobs: Sequence[tuple[int, tuple]],
        consume: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        """Runs one stage for many tables, vectorized over items that share a signature.

        Items that share a model, task and argument shapes run as one call, split
        so that no call's inputs and outputs exceed the worker's call budget.

        Args:
            stage: `fit`, `embed`, `encode` or `decode`.
            jobs: Per item, its table and its arguments after the model
                parameters; host arrays move to the device only with their call.
            consume: Takes each item's outputs as its call finishes, with the
                item's position, instead of holding them all until the end.

        Returns:
            Each item's outputs, in order, or None for each with `consume`.
        """
        groups: dict[tuple, list[int]] = defaultdict(list)
        for index, (table, arguments) in enumerate(jobs):
            spec = self._tables[table]
            groups[(spec.model, spec.num_classes, _signature(arguments))].append(index)
        results: list[Any] = [None] * len(jobs)

        def keep(index: int, item: Any) -> None:
            if consume is None:
                results[index] = item
            else:
                consume(index, item)

        for (model, num_classes, signature), indices in groups.items():
            key = (stage, model, num_classes, signature)
            items = [jobs[index][1] for index in indices]
            self._call_alike(key, items, indices, keep)
        return results

    def _call_alike(
        self,
        key: tuple,
        items: list[tuple],
        indices: list[int],
        consume: Callable[[int, Any], None],
    ) -> None:
        """Runs items of one stage, model, task and signature in as few calls as fit.

        `consume` takes each item's outputs with its index from `indices`.
        """
        stage, model, num_classes, _ = key
        parameters = self._models[model].parameters
        compiled = self._compiled_stage(stage, model, num_classes, len(items[0]))
        estimate = self._estimate(stage, model, num_classes, items[0])
        self._run_items(
            stage,
            key,
            compiled,
            (parameters,),
            items,
            estimate,
            indices,
            consume,
            program=(stage, model, num_classes, len(items[0])),
            buckets=self._buckets(stage),
        )

    def _buckets(self, stage: str) -> tuple[int, ...] | None:
        """The only stacked counts a stage's calls take, or None for every power of two."""
        return FUSED_STACKS if self._stacks and stage == "fit_decode" else None

    def _run_items(
        self,
        stage: str,
        key: tuple,
        compiled: Callable[..., Any],
        shared: tuple,
        items: list[tuple],
        estimate: Callable[[], int],
        indices: Sequence[Any],
        consume: Callable[[Any, Any], None],
        program: tuple | None = None,
        buckets: tuple[int, ...] | None = None,
    ) -> None:
        """Runs alike items through a compiled call in as few calls as fit the budget.

        `consume` takes each item's outputs with its entry in `indices`, and
        `program`, if given, is the `Programs` key `compiled` is kept under.
        Calls take only `buckets`' stacked counts if given, and on an
        accelerator a count not yet compiled compiles on the side.
        """

        def prepare(first: int, stop: int, padded: int) -> Call:
            with self.stats.timed("stack"):
                stacked = _stacked(padded_items(items[first:stop], padded))
            return Call(compiled, shared, stacked, padded, program)

        def taken(first: int, stop: int, outputs: Any) -> None:
            alone = all(leaf.shape[0] == 1 for leaf in jax.tree.leaves(outputs))
            for position in range(stop - first):
                with self.stats.timed("unstack"):
                    item = (
                        _only_item(outputs)
                        if alone
                        else _item(outputs, np.int32(position))
                    )
                consume(indices[first + position], item)

        self._chunks.run(
            stage,
            key,
            len(items),
            self._call_bytes,
            estimate,
            prepare,
            taken,
            self._compile_on_side if self._on_side else None,
            buckets,
        )

    def _compile_on_side(self, call: Call, done: Callable[[bool], None]) -> None:
        """Compiles a call's program for its shapes beside steps, then calls `done`."""
        self.programs.compile_soon(call.function, (*call.shared, *call.items), done)

    def _estimate(
        self, stage: str, model: int, num_classes: int, arguments: tuple
    ) -> Callable[[], int]:
        """What estimates an item's bytes in a call: its inputs and outputs."""

        def measure() -> int:
            function = self._stage(stage, model, num_classes)
            outputs = jax.eval_shape(
                function, self._models[model].parameters, *arguments
            )
            inputs = sum(
                getattr(leaf, "nbytes", 0) for leaf in jax.tree.leaves(arguments)
            )
            return inputs + tree_bytes(outputs)

        def estimate() -> int:
            # Measured by tracing the model, so kept for later processes.
            key = ("estimate", stage, num_classes, repr(kind_signature(arguments)))
            return self._measured(model, key, measure)

        return estimate

    def _warm(self, fit: Fit) -> None:
        """Compiles a newly fit table's page embeds on a thread, before writes need them.

        A write re-embeds the pages it changes, one or a few at a time; on an
        accelerator each new shape compiles for seconds, which the step that
        meets it would otherwise wait for.
        """
        spec = self._tables[fit.table]
        if len(fit.features) == 0:
            return
        features, targets, _ = self._padded(
            fit.features[:1], fit.targets[:1], spec.page_size
        )
        arguments = (spec.state, features, targets)
        # Tables of one model, task and padded shapes share their programs,
        # so only the first of them compiles; many request contexts are.
        marker = ("embeds", spec.model, spec.num_classes, _signature(arguments))
        if self.programs.first(marker):
            self.programs.later(
                lambda: self._compile_embeds(spec.model, spec.num_classes, arguments)
            )

    def _compile_embeds(self, model: int, num_classes: int, arguments: tuple) -> None:
        """Compiles page embeds for each row bucket up to the page size and the first `WARMED_PAGES` page counts, without running one.

        A page with sources embeds only its own rows, padded to their bucket,
        so a step meets the buckets below the page size too. Compiling from
        shapes alone leaves the device to the step thread, so a warm never
        runs beside a step.
        """
        key = ("embed", model, num_classes, _signature(arguments))
        estimate = self._estimate("embed", model, num_classes, arguments)
        size = self._chunks.size(key, self._call_bytes, estimate)
        compiled = self._compiled_stage("embed", model, num_classes, len(arguments))
        state, features, targets = arguments
        page = len(features)
        below = [
            rows for rows in ladder(self._minimum, page.bit_length()) if rows < page
        ]
        for rows in [*below, page]:
            sliced = (state, features[:rows], targets[:rows])
            for items in ladder(1, WARMED_PAGES):
                count = bucket_size(min(items, size), 1)
                stacked = jax.tree.map(
                    lambda value, count=count: _stacked_shape(value, count), sliced
                )
                jitted = cast(jax.stages.Wrapped, compiled)
                jitted.lower(self._models[model].parameters, *stacked).compile()

    def _warmed_queries(self) -> list[int]:
        """Padded query counts compiled ahead: `WARMED_QUERIES` buckets above the smallest."""
        return ladder(self._size(self._minimum + 1), WARMED_QUERIES)

    def _warm_query_embeds(self, batch: QueryBatch) -> None:
        """Compiles a query embed for each of `_warmed_queries` ahead, once per model, task and shapes."""
        spec = self._tables[batch.table]
        if spec.state is None:
            return
        features = np.asarray(batch.features)
        marker = (
            "queries",
            spec.model,
            spec.num_classes,
            _signature((spec.state, features[:0])),
        )
        if not self.programs.first(marker):
            return
        compiled = self._compiled_stage("embed", spec.model, spec.num_classes, 2)
        for more in self._warmed_queries():
            shape = jax.ShapeDtypeStruct((1, more, *features.shape[1:]), features.dtype)
            self.programs.compile_later(
                compiled,
                self._models[spec.model].parameters,
                _stacked_shape(spec.state, 1, stacked=True),
                shape,
            )

    def wait_warm(self) -> None:
        """Waits for every compile started in the background."""
        self.programs.wait()

    def _count(self, stage: str, seconds: float) -> None:
        """Counts a compiled call of a stage, and the seconds it took."""
        # A fit of the state alone counts as a fit.
        stage = "fit" if stage == "state" else stage
        with _COUNTING:
            self.stats.calls[stage] = self.stats.calls.get(stage, 0) + 1
            self.stats.seconds[stage] = self.stats.seconds.get(stage, 0.0) + seconds

    def _compiled(self, stage: str) -> None:
        """Counts a call of a stage whose shape its program had not seen."""
        with _COUNTING:
            self.stats.compiles[stage] = self.stats.compiles.get(stage, 0) + 1

    def _part(self, part: str, seconds: float) -> None:
        """Counts one run of a part of a step; see `WorkerStats.parts`."""
        self.stats.part(part, seconds)

    def _stage(self, stage: str, model: int, num_classes: int) -> Callable[..., Any]:
        """One stage for one table, before vectorizing."""
        entry = self._tabular(model)
        functions = entry.functions
        # Caches as the runtime stores them, or back in float32 for a model
        # that reads them only as its `encode` returned them.
        stored = functions.properties(entry.parameters).stored_cache
        read = (lambda cache: cache) if stored else load_cache
        settings = self._settings

        def stored_dtype(cache: Any) -> Any:
            return (
                cache
                if settings is None
                else to_cache_dtype(cache, settings.cache_dtype)
            )

        def fit(parameters, features, targets, rows, categorical):
            return functions.fit(
                parameters,
                features,
                targets,
                num_classes,
                num_rows=rows,
                categorical=categorical,
            )

        # The state alone: XLA drops the rows `fit` embeds, and each page is
        # embedded once a plan reads it. A column summary over more cells
        # than the call budget takes at once reads a chunk of rows at a time;
        # a fit that embeds its rows takes few enough to read them at once.
        def state(parameters, features, targets, rows, categorical):
            chunk = self._summary_chunk(model, features.shape[-1])
            parameters = with_summary_chunk(parameters, chunk)
            return fit(parameters, features, targets, rows, categorical)[0]

        stages: dict[str, Callable[..., Any]] = {
            "fit": fit,
            "state": state,
            "embed": functions.embed,
            # Cast inside the call, so it holds the cache in one dtype only.
            "encode": lambda parameters, state, rows, targets, count: stored_dtype(
                functions.encode(parameters, state, rows, targets, num_rows=count)
            ),
            "decode": lambda parameters, state, cache, rows: functions.postprocess(
                parameters,
                state,
                functions.decode(parameters, state, read(cache), rows),
            ),
        }

        # A unit fit on its own rows: its state comes with its cache, and its
        # queries embed against it.
        def fit_encode(parameters, features, targets, rows, categorical):
            state, context_rows = fit(parameters, features, targets, rows, categorical)
            encode = stages["encode"]
            return state, encode(parameters, state, context_rows, targets, rows)

        def embed_decode(parameters, state, cache, features):
            rows = functions.embed(parameters, state, features)
            return stages["decode"](parameters, state, cache, rows)

        def fit_decode(parameters, features, targets, rows, categorical, queries):
            state, cache = fit_encode(parameters, features, targets, rows, categorical)
            return state, embed_decode(parameters, state, cache, queries)

        stages["fit_encode"], stages["embed_decode"] = fit_encode, embed_decode
        stages["fit_decode"] = fit_decode
        return stages[stage]

    def _compiled_stage(
        self, stage: str, model: int, num_classes: int, arity: int
    ) -> Callable[..., Any]:
        """A stage vectorized over its arguments, compiled once per model and task."""

        def build() -> Callable[..., Any]:
            function = self._stage(stage, model, num_classes)
            return jax.vmap(function, in_axes=(None,) + (0,) * arity)

        return self.programs.compiled((stage, model, num_classes, arity), build)

    def _padded(
        self, features: np.ndarray, targets: np.ndarray, at_least: int = 1
    ) -> tuple[np.ndarray, np.ndarray, int]:
        count = len(features)
        size = self._size(max(count, at_least))
        with self.stats.timed("pad"):
            return _host_padded(features, size), _host_padded(targets, size), count

    def _keep(self, page: Page, representations: Representations) -> None:
        """Caches a computed page in the host slots the node kept for it, if any."""
        if page.action != "encode":
            return
        values, targets = representations
        # Targets can be views of the whole frame the step came in; a copy keeps
        # the cached page from holding that frame, and its shared memory, alive.
        targets = np.array(targets, copy=True)
        data = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        capacity = len(page.slots) * self._slot_bytes
        if data.size > capacity:
            raise RuntimeError(
                f"a page takes {data.size} bytes, but the node kept {capacity}"
            )
        # Only the page's bytes: a page may fill a fraction of a large slot.
        with self.stats.timed("store"):
            self._host.write_bytes(list(page.slots), data)
        self._pages[page.key] = _Page(
            list(page.slots), values.shape, values.dtype, targets
        )

    def _cached(self, key: CacheKey) -> Representations:
        """A cached page's representations and targets, read from its host slots."""
        page = self._pages[key]
        size = int(np.prod(page.shape)) * page.dtype.itemsize
        data = self._host.read_bytes(page.slots, size)
        return data.view(page.dtype).reshape(page.shape), page.targets

    def _fit(
        self, fits: Sequence[Fit], pages: Sequence[Page]
    ) -> dict[int, Representations]:
        """Fits tables, and splits the rows of fits that fill pages into them."""
        jobs = [
            (
                fit.table,
                (
                    *self._padded(fit.features, fit.targets),
                    self._tables[fit.table].categorical,
                ),
            )
            for fit in fits
        ]
        # A fit filling pages embeds its rows in the call that fits its state;
        # one past the rows a call takes computes the state alone, and each
        # page is embedded once a plan reads it.
        fused = [index for index, fit in enumerate(fits) if fit.pages]
        alone = [index for index, fit in enumerate(fits) if not fit.pages]
        for index, state in zip(
            alone, self._call("state", [jobs[index] for index in alone]), strict=True
        ):
            self._tables[fits[index].table].state = state
            self._warn_crowded(fits[index].table, state)
            self._warm(fits[index])
        filled: dict[int, Representations] = {}
        for index, (state, representations) in zip(
            fused, self._call("fit", [jobs[index] for index in fused]), strict=True
        ):
            fit = fits[index]
            self._tables[fit.table].state = state
            self._warn_crowded(fit.table, state)
            self._warm(fit)
            # Split on the host: slicing on the device compiles once per page
            # length, and pages change length with every write.
            with self.stats.timed("fetch"):
                on_host = np.asarray(representations)
            offset = 0
            for index in fit.pages:
                count = len(pages[index].rows)
                targets = np.asarray(fit.targets[offset : offset + count])
                filled[index] = (on_host[offset : offset + count], targets)
                self._keep(pages[index], filled[index])
                offset += count
        return filled

    def _warn_crowded(self, table: int, state: Any) -> None:
        """Logs once per table that categorical columns held more categories than kept.

        Args:
            table: The table a state was fit for, on the whole table or a unit.
            state: The fitted state.
        """
        entry = self._tables[table]
        count = self._models[entry.model].functions.crowded_categories
        if entry.crowded or entry.categorical is None or count is None:
            return
        crowded = count(state)
        if crowded:
            entry.crowded = True
            columns = "column holds" if crowded == 1 else "columns hold"
            logger.warning(
                "table %d: %d categorical %s more categories than the model "
                "keeps, so their ranks are interpolated",
                table,
                crowded,
                columns,
            )

    def _embed_pages(
        self, pages: Sequence[Page], filled: dict[int, Representations]
    ) -> dict[int, Representations]:
        """Reads cached pages and embeds the rest, one vectorized call per signature.

        A page with sources copies their rows, representations and targets,
        and embeds only its own. Those pad to their own bucket, not the page
        size: a page written a row at a time then embeds a few rows, or none.
        """
        found: dict[int, Representations] = {}
        jobs: list[tuple[int, tuple]] = []
        embedding: list[tuple[int, list[int]]] = []
        partial: dict[int, Representations] = {}
        for index, page in enumerate(pages):
            if index in filled:
                continue
            if page.action == "reuse":
                found[index] = self._cached(page.key)
                continue
            if page.features is None or page.targets is None:
                raise RuntimeError(
                    f"a page of table {page.table} to embed came without its rows"
                )
            missing = list(range(len(page.rows)))
            if page.copies:
                partial[index] = self._copied(page)
                copied = {to for _, positions in page.copies for _, to in positions}
                missing = [at for at in missing if at not in copied]
            if not missing:
                continue
            at_least = 1 if page.copies else self._tables[page.table].page_size
            features, targets, _ = self._padded(page.features, page.targets, at_least)
            jobs.append(
                (page.table, (self._tables[page.table].state, features, targets))
            )
            embedding.append((index, missing))
        for (index, missing), representations in zip(
            embedding, self._call("embed", jobs), strict=True
        ):
            page = pages[index]
            with self.stats.timed("fetch"):
                embedded = np.asarray(representations)[: len(missing)]
            if index in partial:
                values, targets = partial[index]
                values[missing] = embedded
                targets[missing] = np.asarray(page.targets)
            else:
                partial[index] = (embedded, np.asarray(page.targets))
            self.stats.rows_embedded += len(missing)
        for index, computed in partial.items():
            self._keep(pages[index], computed)
            found[index] = computed
        return found

    def _copied(self, page: Page) -> Representations:
        """A page's representations and targets with its sources' rows copied in."""
        values: np.ndarray | None = None
        targets: np.ndarray | None = None
        for source, positions in page.copies:
            held, labels = self._cached(source)
            if values is None or targets is None:
                values = np.empty((len(page.rows), *held.shape[1:]), held.dtype)
                targets = np.empty((len(page.rows), *labels.shape[1:]), labels.dtype)
            at, to = (list(part) for part in zip(*positions, strict=True))
            values[to] = held[at]
            targets[to] = labels[at]
            self.stats.rows_copied += len(positions)
        if values is None or targets is None:
            raise RuntimeError(f"a page of table {page.table} copies from no source")
        return values, targets

    @staticmethod
    def _tables_rows(
        pages: Sequence[Page], computed: dict[int, Representations]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each table's page rows sorted, with their representations and targets.

        Built once per step, so units gather from them without joining pages
        again. They join on the host: on the device, every new number of rows
        would compile its own join and gather, and writes change it each time.
        """
        by_table: dict[int, list[int]] = defaultdict(list)
        for index, page in enumerate(pages):
            by_table[page.table].append(index)
        joined = {}
        for table, indices in by_table.items():
            rows = np.sort(
                np.concatenate([np.asarray(pages[index].rows) for index in indices])
            )
            # Each page's rows go straight to their place, so the step holds
            # one copy of the representations beside the pages'.
            first_values, first_targets = (
                np.asarray(part) for part in computed[indices[0]]
            )
            values = np.empty((len(rows), *first_values.shape[1:]), first_values.dtype)
            targets = np.empty(
                (len(rows), *first_targets.shape[1:]), first_targets.dtype
            )
            for index in indices:
                at = np.searchsorted(rows, np.asarray(pages[index].rows))
                values[at], targets[at] = computed[index]
            joined[table] = (rows, values, targets)
        return joined

    def _encode(
        self,
        units: Sequence[Unit],
        pages: Sequence[Page],
        computed: dict[int, Representations],
    ) -> Transient:
        """Encodes the units that are not cached; returns those there was no room to keep.

        A unit with no room to keep is encoded only by the decode that reads
        it, so a step never holds every such context at once.
        """
        transient: Transient = {}
        waiting = [
            index
            for index, unit in enumerate(units)
            if unit.action not in ("reuse", "fuse")
        ]
        self.stats.reused += sum(unit.action == "reuse" for unit in units)
        self.stats.rows_reused += sum(
            len(unit.rows) for unit in units if unit.action == "reuse"
        )
        joined = self._tables_rows(pages, computed) if waiting else {}
        jobs: dict[str, list[tuple[int, tuple]]] = defaultdict(list)
        group: dict[str, list[int]] = defaultdict(list)
        for index in waiting:
            unit = units[index]
            self._fits_device(unit)
            if self._tables[unit.table].own is not None:
                stage, arguments = "fit_encode", self._own_arguments(unit)
            else:
                stage, arguments = "encode", self._gathered(unit, joined)
            if unit.action == "transient":
                transient[index] = _Deferred(
                    unit.table, arguments, len(unit.rows), stage
                )
                continue
            jobs[stage].append((unit.table, arguments))
            group[stage].append(index)

        # Each context goes to its slots as its call finishes, so a step never
        # holds all it encodes beside the pool.
        def keeper(stage: str) -> Callable[[int, Any], None]:
            def keep(position: int, outputs: Any) -> None:
                unit = units[group[stage][position]]
                state, cache = outputs if stage == "fit_encode" else (None, outputs)
                if state is not None:
                    self._warn_crowded(unit.table, state)
                self.stats.encoded += 1
                self.stats.rows_encoded += len(unit.rows)
                table = self._tables[unit.table]
                layout = self._layouts[table.model, table.features]
                self._store(
                    unit.key,
                    unit.slots,
                    layout,
                    len(unit.rows),
                    cache,
                    table.own,
                    state,
                )

            return keep

        for stage, items in jobs.items():
            self._call(stage, items, consume=keeper(stage))
        return transient

    def _gathered(
        self, unit: Unit, joined: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> tuple:
        """A unit's encode arguments, its rows' representations gathered from pages."""
        rows, values, targets = joined[unit.table]
        wanted = np.asarray(unit.rows)
        positions = np.searchsorted(rows, wanted)
        if np.any(positions >= len(rows)) or np.any(rows[positions] != wanted):
            raise RuntimeError(
                f"a unit of table {unit.table} reads rows that are in none "
                "of the step's pages"
            )
        size = self._size(len(unit.rows))
        with self.stats.timed("pad"):
            return (
                self._tables[unit.table].state,
                _host_padded(values[positions], size),
                _host_padded(targets[positions], size),
                len(unit.rows),
            )

    def _own_arguments(self, unit: Unit) -> tuple:
        """A unit's fit arguments, from the rows it came with."""
        if unit.features is None or unit.targets is None:
            raise RuntimeError(
                f"a unit of table {unit.table} to fit came without its rows"
            )
        size = self._size(len(unit.rows))
        with self.stats.timed("pad"):
            return (
                _host_padded(unit.features, size),
                _host_padded(unit.targets, size),
                len(unit.rows),
                self._tables[unit.table].categorical,
            )

    def _fits_device(self, unit: Unit) -> None:
        """Refuses a unit whose encode, or whose context and decode, would not fit the device.

        Beside the pool, the device must hold the context's encode: its inputs,
        temporaries and cache, which the call's lone item is taken from in
        place. It must also hold the working memory of decoding the
        smallest chunk of queries against the context. A kept context's decode
        reads it, and any own state, from the pool into that working memory; a
        transient context's decode takes a copy of it into its call, beside
        the one held. On an accelerator, XLA's autotuner also needs headroom.
        Otherwise the request would fail for want of memory, after a long fit,
        with nothing said about what to do instead.

        Raises:
            ValueError: Naming the strategies that read fewer rows.
        """
        table = self._tables[unit.table]
        per_row = self.profile(table.model, table.features).cache_bytes_per_row
        size = self._size(len(unit.rows))
        context = self._context_bytes(unit)
        decode = self.decode_profile(table.model)
        working = decode.working_bytes(per_row, size)
        measured = self.encode_bytes(
            table.model, table.features, size, table.own is not None
        )
        encode = 0 if measured is None else measured
        if unit.action == "transient":
            needed = max(encode, 2 * context + working)
        else:
            # A unit's own state is read from its slots and unpacked beside it.
            state = 0 if table.own is None else 2 * table.own.slots * self._slot_bytes
            needed = max(encode, working + state)
        if not self.budget.cpu:
            needed += max(encode, working) // AUTOTUNE_SHARE
        room = self.budget.device - self._pool_bytes
        if needed > room:
            encoding = (
                "" if measured is None else f"encoding it {encode >> 20} MiB and "
            )
            raise ValueError(
                f"a context of {len(unit.rows)} rows takes {context >> 20} MiB as the "
                f"model caches it, {encoding}decoding "
                f"{decode.queries} queries against it {working >> 20} MiB, which "
                f"needs {needed >> 20} MiB, but the device has {room >> 20} MiB "
                "beside the contexts it keeps; use a strategy that reads fewer "
                "rows, such as neighbors, or a device with more memory"
            )

    def _context_bytes(self, unit: Unit) -> int:
        """Bytes of a unit's context as the model caches it, padding and own state included."""
        table = self._tables[unit.table]
        profile = self.profile(table.model, table.features)
        own = 0 if table.own is None else table.own.slots * self._slot_bytes
        return self._size(len(unit.rows)) * profile.cache_bytes_per_row + own

    def _store(
        self,
        key: CacheKey,
        slots: Sequence[int],
        layout: CacheLayout,
        rows: int,
        cache: Any,
        own: StateLayout | None = None,
        state: Any = None,
    ) -> None:
        """Packs an encoded context of `rows` layout rows, and any own state after it, into the device slots the node assigned it."""
        slots = list(slots)
        packed = layout.packed_slots(cache)
        own_data = None if own is None else pack_state(own, state)
        used = packed + (0 if own_data is None else own_data.shape[0])
        # The node keeps room for the table's state, with or without
        # categorical columns; the state may take fewer whole slots.
        written = slots if own is None else slots[:used]
        if used != len(written):
            raise RuntimeError(
                f"a context packs into {used} slots, but the node gave {len(slots)}"
            )
        with self.stats.timed("store"):
            self._device.write_cache(slots[:packed], layout, cache)
            if own_data is not None:
                self._device.write(slots[packed:used], own_data)
        self._contexts[key] = Context(layout, rows, slots)

    def _decode(
        self,
        batches: Sequence[QueryBatch],
        units: Sequence[Unit],
        transient: Transient,
    ) -> dict[int, np.ndarray]:
        """Decodes every query against the unit it reads.

        A step's requests to one table decode as one batch, so their queries
        embed in one call and queries reading the same context read it once.
        Cached contexts are read straight from their slots, every context of a
        model and task in one call. Contexts there was no room to keep are
        encoded and decoded as dense arrays.
        """
        joined, parts = merged(batches)
        predictions = self._decode_batches(joined, units, transient)
        return {
            request: predictions[index][start:stop]
            for index, owned in enumerate(parts)
            if index in predictions
            for request, start, stop in owned
        }

    def _decode_fused(
        self, batches: Sequence[QueryBatch], units: Sequence[Unit]
    ) -> dict[int, np.ndarray]:
        """Fits, encodes and decodes each fused unit and its queries in one call.

        A fused unit is a context one request brought: no page or slot keeps
        it, and its table keeps only the state, so a later step that reads the
        context again encodes it from pages. Units of one model, task and
        padded shapes share their calls.
        """
        if not batches:
            return {}
        joined, parts = merged(batches)
        jobs = []
        kept = []
        for batch, owned in zip(joined, parts, strict=True):
            unit = units[batch.units[0]]
            if any(units[index] is not unit for index in batch.units):
                raise RuntimeError(
                    f"queries of table {batch.table} read more than its fused unit"
                )
            with self.stats.timed("pad"):
                queries = _host_padded(batch.features, self._size(len(batch.units)))
            arguments = (*self._own_arguments(unit), queries)
            if self._defers(unit, arguments):
                continue
            self._fits_device(unit)
            jobs.append((batch.table, arguments))
            kept.append((batch, owned))
        predictions: dict[int, np.ndarray] = {}
        if self._stacks:
            outputs = self._call("fit_decode", jobs)
        else:
            outputs = self._alone("fit_decode", jobs)
        for (batch, owned), (state, values) in zip(kept, outputs, strict=True):
            unit = units[batch.units[0]]
            self._tables[batch.table].state = state
            self._warn_crowded(batch.table, state)
            self.stats.encoded += 1
            self.stats.transient += 1
            self.stats.rows_encoded += len(unit.rows)
            with self.stats.timed("fetch"):
                values = np.asarray(values)[: len(batch.units)]
            for request, start, stop in owned:
                predictions[request] = values[start:stop]
        return predictions

    def _defers(self, unit: Unit, arguments: tuple) -> bool:
        """Whether a fused unit waits for its program, readied beside steps.

        On an accelerator a new shape's program compiles for tens of
        seconds, and its first unit's measures take as long again, which
        every other table's steps would wait for. Such a unit's table is
        deferred instead, under its program's key, which `announce` names
        once the program is ready, for the server to plan it again.
        """
        if not self._on_side or self.announce is None:
            return False
        table = self._tables[unit.table]
        key = ("fit_decode", table.model, table.num_classes, _signature(arguments))
        if self._chunks.ran(key):
            return False
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        name = f"{self.programs.drops}:{digest}"
        if name in self._unpreparable:
            return False
        self.deferred[unit.table] = name
        if name not in self._preparing:
            self._preparing.add(name)
            self.programs.soon(lambda: self._prepare(unit, key, arguments, name))
        return True

    def _prepare(self, unit: Unit, key: tuple, arguments: tuple, name: str) -> None:
        """Measures a deferred unit and compiles its program for one, from shapes, then announces its key."""
        stage, model, num_classes, _ = key
        program = (stage, model, num_classes, len(arguments))
        try:
            self._fits_device(unit)
            compiled = self._compiled_stage(*program)
            shared = (self._models[model].parameters,)
            items = _shapes(_stacked([arguments]))
            cast(jax.stages.Wrapped, compiled).lower(*shared, *items).compile()
            self._chunks.mark(key, 1)
            self._seen(Call(compiled, shared, items, 1, program))
        except Exception as error:  # noqa: BLE001
            logger.warning(
                "could not ready a program beside steps (%s); the step that "
                "next needs it compiles it",
                error,
            )
            self._unpreparable.add(name)
        finally:
            self._preparing.discard(name)
            if self.announce is not None:
                self.announce([name])

    def _alone(self, stage: str, jobs: Sequence[tuple[int, tuple]]) -> list[Any]:
        """Runs each job in a call of its own, up to a core's worth side by side.

        Outputs come back in the jobs' order. Every call ends before this
        returns, and a failure raises the first failed job's error, as when
        they run in turn, so `run_apart` still fails only its table.
        """
        if len(jobs) < 2 or self._side_by_side < 2:
            return [self._call(stage, [job])[0] for job in jobs]
        if self._alongside is None:
            self._alongside = ThreadPoolExecutor(
                self._side_by_side, thread_name_prefix="gf-alone"
            )
        calls: list[Future[Any]] = [
            self._alongside.submit(lambda job=job: self._call(stage, [job])[0])
            for job in jobs
        ]
        finished(calls)
        return [call.result() for call in calls]

    def _decode_batches(
        self,
        batches: Sequence[QueryBatch],
        units: Sequence[Unit],
        transient: Transient,
    ) -> dict[int, np.ndarray]:
        """Decodes batches whose requests are their positions; see `_decode`.

        A table that summarizes each context embeds its queries against their
        unit's own state, in the decode, so each decoding reads one unit.
        """
        features = []
        for batch in batches:
            with self.stats.timed("pad"):
                padded = _host_padded(batch.features, self._size(len(batch.units)))
            with self.stats.timed("dispatch"):
                features.append(jnp.asarray(padded))
        fit_own = [self._tables[batch.table].own is not None for batch in batches]
        embedded = iter(
            self._call(
                "embed",
                [
                    (batch.table, (self._tables[batch.table].state, rows))
                    for batch, rows, mine in zip(
                        batches, features, fit_own, strict=True
                    )
                    if not mine
                ],
            )
        )
        for batch in batches:
            self._warm_query_embeds(batch)
        dense: list[_Dense] = []
        paged: dict[tuple, list[Decoding]] = defaultdict(list)
        for batch, raw, mine in zip(batches, features, fit_own, strict=True):
            rows = raw if mine else next(embedded)
            groups: dict[int, list[int]] = defaultdict(list)
            for query, unit in enumerate(batch.units):
                groups[unit].append(query)
            table = self._tables[batch.table]
            layout = self._layouts[table.model, table.features]
            together: list[Group] = []
            for unit, queries in groups.items():
                if unit in transient:
                    dense.append(_Dense(batch, rows, unit, queries))
                else:
                    together.append((unit, queries))
            # Only states of one structure stack: a table with categorical
            # columns can keep more state than one without.
            structure = jax.tree.structure(table.state)
            key = (table.model, table.num_classes, layout, table.own, structure)
            packs = (
                [[one] for one in together] if mine else self._packed(together, units)
            )
            for groups_of in packs:
                paged[key].append(self._decoding(batch, rows, groups_of, units))
        outputs = self._decode_dense(dense, units, transient)
        for (model, num_classes, layout, own, _), decodings in paged.items():
            for same in alike(decodings, layout.slots_per_row, self._size):
                outputs += self._decode_paged(model, num_classes, layout, same, own)
        with self.stats.timed("fetch"):
            fetched = jax.device_get([values for _, values in outputs])
        predictions: dict[int, np.ndarray] = {}
        with self.stats.timed("scatter"):
            parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for ((batch, queries), _), values in zip(outputs, fetched, strict=True):
                values = np.asarray(values)
                if batch.request not in predictions:
                    shape = (len(batch.units), values.shape[-1])
                    predictions[batch.request] = np.zeros(shape, dtype=values.dtype)
                at = np.asarray(queries, dtype=np.intp)
                for start in range(0, len(at), SCATTER_ROWS):
                    stop = start + SCATTER_ROWS
                    parts.append(
                        (predictions[batch.request], at[start:stop], values[start:stop])
                    )
            self._scatter(parts)
        return predictions

    def _scatter(self, parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
        """Copies each part's values to its rows of its output, side by side on the worker's cores."""

        def put(part: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
            output, rows, values = part
            output[rows] = values[: len(rows)]

        if len(parts) < 2 or self._scatter_threads < 2:
            for part in parts:
                put(part)
            return
        if self._scatterers is None:
            self._scatterers = ThreadPoolExecutor(
                self._scatter_threads, thread_name_prefix="gf-scatter"
            )
        for _ in self._scatterers.map(put, parts):
            pass

    def _decode_dense(
        self, jobs: list[_Dense], units: Sequence[Unit], transient: Transient
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Decodes queries against units put off until now, a window of contexts at a time.

        Encoding each such unit makes a context, so encoding every one of a
        step first would hold them all at once.
        """
        outputs: list[tuple[tuple[QueryBatch, list[int]], Any]] = []
        window: list[_Dense] = []
        held = 0
        for job in jobs:
            window.append(job)
            held += self._context_bytes(units[job.unit])
            if held >= self._window_bytes:
                outputs += self._decode_window(window, transient)
                window, held = [], 0
        return outputs + self._decode_window(window, transient)

    def _decode_window(
        self, jobs: list[_Dense], transient: Transient
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Decodes some dense queries, encoding the units they read first."""
        encoded = self._encode_deferred([job.unit for job in jobs], transient)
        kinds: dict[tuple, list[_Dense]] = defaultdict(list)
        for job in jobs:
            table = self._tables[job.batch.table]
            kinds["dense", table.model, table.num_classes, table.features].append(job)
        outputs: list[tuple[tuple[QueryBatch, list[int]], Any]] = []
        for key, same in kinds.items():
            outputs += self._decode_parts(key, same, encoded)
        return outputs

    def _decode_parts(
        self, key: tuple, jobs: list[_Dense], encoded: dict[int, tuple[Any, Any]]
    ) -> list[tuple[tuple[QueryBatch, list[int]], Any]]:
        """Decodes dense queries of one model, task and width against their contexts.

        A unit fit on its own rows comes with its state, which its queries'
        features embed against. A context too large to decode all its queries
        at once decodes them in parts, halved until they fit, down to the
        smallest chunk; later calls start at that size.
        """
        limit = self._chunks.parts(key)
        parts = [part for job in jobs for part in job.split(limit)]
        stages: dict[str, list[int]] = defaultdict(list)
        items: list[tuple[int, tuple]] = []
        for index, part in enumerate(parts):
            own, cache = encoded[part.unit]
            rows = query_rows(part.rows, part.queries, self._size(len(part.queries)))
            if own is None:
                stages["decode"].append(index)
                state = self._tables[part.batch.table].state
            else:
                stages["embed_decode"].append(index)
                state = own
            items.append((part.batch.table, (state, cache, rows)))
        done: dict[int, Any] = {}

        def keeper(indices: list[int]) -> Callable[[int, Any], None]:
            def keep(position: int, values: Any) -> None:
                done[indices[position]] = values

            return keep

        try:
            for stage, indices in stages.items():
                self._call(
                    stage, [items[index] for index in indices], consume=keeper(indices)
                )
        except jax.errors.JaxRuntimeError as error:
            rest = [part for index, part in enumerate(parts) if index not in done]
            most = max(len(part.queries) for part in rest)
            if not out_of_memory(error) or most <= self._size(1):
                raise
            # One context's queries alone ran out of memory: they go in halves,
            # from here on.
            self._chunks.split(key, max(self._size(1), most // 2))
            finished = [
                ((part.batch, part.queries), done[index])
                for index, part in enumerate(parts)
                if index in done
            ]
            return finished + self._decode_parts(key, rest, encoded)
        return [
            ((part.batch, part.queries), done[index])
            for index, part in enumerate(parts)
        ]

    def _encode_deferred(
        self, indices: list[int], transient: Transient
    ) -> dict[int, tuple[Any, Any]]:
        """Encodes the put-off units among `indices`, once each, in shared calls.

        Returns:
            Each unit's own state, or None for one that gathered its rows from
            pages, and its cache.
        """
        wanted = list(dict.fromkeys(indices))
        deferred = [transient[unit] for unit in wanted]
        outputs = self._staged([(d.stage, d.table, d.arguments) for d in deferred])
        encoded: dict[int, tuple[Any, Any]] = {}
        for unit, put_off, output in zip(wanted, deferred, outputs, strict=True):
            state, cache = output if put_off.stage == "fit_encode" else (None, output)
            if state is not None:
                self._warn_crowded(put_off.table, state)
            if not put_off.encoded:
                put_off.encoded = True
                self.stats.encoded += 1
                self.stats.transient += 1
                self.stats.rows_encoded += put_off.rows
            encoded[unit] = (state, cache)
        return encoded

    def _staged(self, jobs: Sequence[tuple[str, int, tuple]]) -> list[Any]:
        """Runs jobs of several stages, one `_call` per stage, with outputs in order."""
        by_stage: dict[str, list[int]] = defaultdict(list)
        for index, (stage, _, _) in enumerate(jobs):
            by_stage[stage].append(index)
        outputs: list[Any] = [None] * len(jobs)
        for stage, indices in by_stage.items():
            done = self._call(stage, [jobs[index][1:] for index in indices])
            for index, output in zip(indices, done, strict=True):
                outputs[index] = output
        return outputs

    def _packed(self, groups: list[Group], units: Sequence[Unit]) -> list[list[Group]]:
        """Query groups packed into decodings that read few slots each; see `packed`."""
        slots = {unit: len(self._contexts[units[unit].key].slots) for unit, _ in groups}
        return packed(groups, slots, self._window_bytes // self._slot_bytes)

    def _decoding(
        self, batch: QueryBatch, rows: Any, groups: list[Group], units: Sequence[Unit]
    ) -> Decoding:
        """One decoding of a batch's queries against the units they read."""
        contexts = {unit: self._contexts[units[unit].key] for unit, _ in groups}
        padded = self._size(sum(len(members) for _, members in groups))
        return decoding_of(batch, rows, groups, contexts, padded)

    def _decode_paged(
        self,
        model: int,
        num_classes: int,
        layout: CacheLayout,
        decodings: list[Decoding],
        own: StateLayout | None = None,
    ) -> list[tuple[tuple[QueryBatch, list[int]], np.ndarray]]:
        """Decodes contexts read from their slots, as few calls as the budget allows.

        A context too large to decode all its queries at once decodes them in
        parts, halved until they fit, down to the smallest chunk; later calls
        start at that size. With `own`, how units keep their own states, each
        decoding reads one unit, whose state is read from its slots and embeds
        the queries' features.
        """
        key = ("paged", model, num_classes, layout, own)
        limit = self._chunks.parts(key)
        if limit is not None:
            decodings = [part for d in decodings for part in split(d, limit)]
        shape = paged_shape(decodings, layout, self._size)
        signature = (key, shape)
        program = self._paged_key(
            model, num_classes, layout, shape.masked, shape.in_order, own is not None
        )
        compiled = self._program(program)
        parameters = self._models[model].parameters
        results: list[tuple[tuple[QueryBatch, list[int]], np.ndarray]] = []
        # The latest call alone, so earlier chunks' arguments are freed.
        latest: list[Call] = []

        def prepare(first: int, stop: int, padded: int) -> Call:
            with self.stats.timed("pad"):
                tables, inputs = paged_inputs(
                    decodings[first:stop], layout, shape, padded
                )
            index, blocks, *rest = inputs
            if own is not None:
                states = self._own_states(decodings[first:stop], own, padded)
                index = np.arange(len(index), dtype=np.int32)
            else:
                with self.stats.timed("stack"):
                    states = _stacked(
                        padded_items(
                            [self._tables[table].state for table in tables], padded
                        )
                    )
            # Rows come straight from the pool into the decode's cache, so it
            # holds neither the slots' bytes whole nor a copy of the pool.
            source, blocks = self._device.source(blocks)
            arguments = (states, index, blocks, *rest)
            latest[:] = [
                Call(compiled, (parameters, source), arguments, len(index), program)
            ]
            return latest[0]

        def taken(first: int, stop: int, values: Any) -> None:
            if self.programs.first(("decodes", signature)):
                call = latest[0]
                self._warm_decodes(compiled, call.shared, call.items, shape.queries)
            for position, decoding in enumerate(decodings[first:stop]):
                with self.stats.timed("unstack"):
                    item = _item(values, np.int32(position))
                results.append(((decoding.batch, decoding.queries), item))

        # A context and the working copy beside it, and the queries' rows; and
        # a unit's own state, as its slots' bytes and unpacked.
        width = decodings[0].rows.shape[-1]
        estimate = 2 * shape.context_rows * layout.row_bytes + shape.queries * width * 4
        if own is not None:
            estimate += 2 * own.slots * self._slot_bytes
        try:
            self._chunks.run(
                "decode",
                signature,
                len(decodings),
                self._call_bytes,
                lambda: estimate,
                prepare,
                taken,
                self._compile_on_side if self._on_side else None,
            )
        except jax.errors.JaxRuntimeError as error:
            # One decoding alone ran out of memory: its queries go in halves,
            # from here on.
            most = len(decodings[len(results)].queries)
            if not out_of_memory(error) or most <= self._size(1):
                raise
            self._chunks.split(key, max(self._size(1), most // 2))
            rest = decodings[len(results) :]
            return results + self._decode_paged(model, num_classes, layout, rest, own)
        return results

    def _own_states(
        self, decodings: list[Decoding], own: StateLayout, count: int
    ) -> Any:
        """The own states of the units `decodings` read, one each, from their slots, padded to `count`."""
        contexts = padded_items([decoding.contexts[0] for decoding in decodings], count)
        slots = []
        for context in contexts:
            start = context.layout.slots(self._size(context.rows))
            slots += context.slots[start : start + own.slots]
        with self.stats.timed("fetch"):
            data = self._device.read(slots).reshape(len(contexts), own.slots, -1)
        return unpack_states(own, data)

    def _warm_decodes(
        self,
        compiled: Callable[..., Any],
        shared: tuple,
        arguments: tuple,
        queries: int,
    ) -> None:
        """Compiles a paged decode ahead for one context and each larger of `_warmed_queries`."""

        def one(value: Any, queries: int | None = None) -> jax.ShapeDtypeStruct:
            shape = list(np.shape(value))
            shape[0] = 1
            if queries is not None:
                shape[1] = queries
            return jax.ShapeDtypeStruct(
                tuple(shape),
                value.dtype,
                weak_type=bool(getattr(value, "weak_type", False)),
            )

        parameters, source = shared
        states, index, blocks, positions, counts, rows, *reads = arguments
        for more in self._warmed_queries():
            if more > queries:
                self.programs.compile_later(
                    compiled,
                    parameters,
                    source,
                    jax.tree.map(one, states),
                    one(index),
                    one(blocks),
                    one(positions),
                    one(counts),
                    one(rows, more),
                    *[one(read, more) for read in reads],
                )

    @staticmethod
    def _paged_key(
        model: int,
        num_classes: int,
        layout: CacheLayout,
        masked: bool,
        in_order: bool,
        own: bool,
    ) -> tuple:
        """The `Programs` key of a model and task's paged decodes for a layout and way of reading."""
        return ("paged", model, num_classes, layout, masked, in_order, own)

    def _compiled_paged(
        self,
        model: int,
        num_classes: int,
        layout: CacheLayout,
        masked: bool,
        in_order: bool,
        own: bool,
    ) -> Callable[..., Any]:
        """Paged decodes of a model and task, compiled once per layout and way of reading."""

        def build() -> Callable[..., Any]:
            entry = self._tabular(model)
            functions = entry.functions
            stored = functions.properties(entry.parameters).stored_cache
            return paged_decode(functions, stored, layout, masked, in_order, own)

        key = self._paged_key(model, num_classes, layout, masked, in_order, own)
        return self.programs.compiled(key, build)

    def _program(self, key: tuple) -> Callable[..., Any]:
        """The compiled function a `Call.program` key names: a stage or a paged decode."""
        if key[0] == "paged":
            return self._compiled_paged(*key[1:])
        stage, model, num_classes, arity = key
        return self._compiled_stage(stage, model, num_classes, arity)

    def _forecast(
        self, units: Sequence[SeriesUnit], forecasts: Sequence[Forecast]
    ) -> dict[int, np.ndarray]:
        """Forecasts every series from the unit it reads.

        Units a reusable model may keep encode once, in one call per model and
        shape, and pack into their slots with the patches first; their
        forecasts decode from the slots, every group of a model, layout and
        horizon in one call. Units forecast whole, and every unit of a model
        that keeps no cache, run the model's `forecast` with the group's known
        future, one call per model, shape and horizon. Forecasts of one unit
        over one padded horizon share a call, whichever members they are of.

        Returns:
            Each request's forecasts, `(series, horizon, quantiles)`.
        """
        groups: dict[int, SeriesGroup] = {}
        for index, unit in enumerate(units):
            properties = self.series_properties(self._tables[unit.table].model)
            values, observed, roles = _grid_of(unit)
            rows, steps = buckets(
                values.shape[0], values.shape[1], properties, self._minimum
            )
            groups[index] = laid_group(values, observed, roles, rows, steps)
            self._fits_device_series(self._tables[unit.table].model, rows, steps)
            if unit.action == "transient":
                self.stats.transient += 1
        self._encode_series(units, groups)
        whole: dict[tuple[int, int], list[int]] = defaultdict(list)
        cached: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, forecast in enumerate(forecasts):
            unit = units[forecast.unit]
            properties = self.series_properties(self._tables[unit.table].model)
            horizon = horizon_bucket(forecast.horizon, properties)
            if unit.action == "transient" or not properties.reusable:
                whole[forecast.unit, horizon].append(index)
            else:
                cached[forecast.unit, horizon].append(index)
        results: dict[int, np.ndarray] = {}

        def keep(item: tuple[int, int], outputs: Any) -> None:
            unit, _ = item
            # A member's rows are its target's, then one per feature column.
            features = self._tables[units[unit].table].features
            outputs = np.asarray(outputs)
            for index in whole.get(item, []) + cached.get(item, []):
                forecast = forecasts[index]
                row = forecast.target * (1 + features)
                results[index] = outputs[row, : forecast.horizon]

        self._forecast_whole(units, forecasts, groups, whole, keep)
        self._decode_series(units, cached, keep)
        collected: dict[int, list[np.ndarray]] = defaultdict(list)
        for index, forecast in enumerate(forecasts):
            collected[forecast.request].append(results[index])
        return {request: np.stack(rows) for request, rows in collected.items()}

    def _forecast_whole(
        self,
        units: Sequence[SeriesUnit],
        forecasts: Sequence[Forecast],
        groups: dict[int, SeriesGroup],
        whole: dict[tuple[int, int], list[int]],
        keep: Callable[[tuple[int, int], Any], None],
    ) -> None:
        """Runs `forecast` for the groups forecast whole, alike ones together.

        Each unit's group takes the known future its forecasts share over
        the padded horizon.
        """
        alike: dict[tuple, list[tuple[tuple[int, int], SeriesGroup]]] = defaultdict(
            list
        )
        for (unit, horizon), indices in whole.items():
            model = self._tables[units[unit].table].model
            padded = groups[unit]
            future, future_observed = _future_of(forecasts[indices[0]])
            # A table without known columns sends no future rows.
            known = future.shape[0] > 0
            group = laid_group(
                *_grid_of(units[unit]),
                padded.values.shape[0],
                padded.values.shape[1],
                future if known else None,
                future_observed if known else None,
                horizon if known else None,
            )
            alike[(model, horizon, _signature((group,)))].append(
                ((unit, horizon), group)
            )
        for (model, horizon, signature), members in alike.items():
            entry = self._series(model)
            forecast = entry.functions.forecast

            def build(forecast: Any = forecast, horizon: int = horizon) -> Any:
                return jax.vmap(lambda p, g: forecast(p, g, horizon), in_axes=(None, 0))

            compiled = self.programs.compiled(("forecast", model, horizon), build)
            items = [(group,) for _, group in members]

            def estimate(
                items: list[tuple] = items, entry: Any = entry, horizon: int = horizon
            ) -> int:
                shape = jax.eval_shape(
                    lambda p, g: entry.functions.forecast(p, g, horizon),
                    entry.parameters,
                    items[0][0],
                )
                return tree_bytes(items[0]) + tree_bytes(shape)

            self._run_items(
                "forecast",
                ("forecast", model, horizon, signature),
                compiled,
                (entry.parameters,),
                items,
                estimate,
                [item for item, _ in members],
                keep,
            )

    def _encode_series(
        self, units: Sequence[SeriesUnit], groups: dict[int, SeriesGroup]
    ) -> None:
        """Encodes the series units to keep, and packs each into its slots, patches first; extends the units grown from a kept one."""
        alike: dict[tuple, list[int]] = defaultdict(list)
        extensions = []
        for index, unit in enumerate(units):
            model = self._tables[unit.table].model
            if unit.action == "reuse":
                self.stats.reused += 1
                continue
            if unit.action == "extend":
                extensions.append(index)
                continue
            if unit.action != "encode" or not self.series_properties(model).reusable:
                continue
            alike[(model, _signature((groups[index],)))].append(index)
        self._extend_series(units, groups, extensions)
        for (model, signature), indices in alike.items():
            entry = self._series(model)
            encode = entry.functions.encode
            assert encode is not None
            compiled = self.programs.compiled(
                ("series_encode", model),
                lambda encode=encode: jax.vmap(encode, in_axes=(None, 0)),
            )
            items = [(groups[index],) for index in indices]

            def estimate(
                items: list[tuple] = items,
                encode: Any = encode,
                entry: _SeriesModel = entry,
            ) -> int:
                shape = jax.eval_shape(encode, entry.parameters, items[0][0])
                return tree_bytes(items[0]) + 2 * tree_bytes(shape)

            def keep(index: int, cache: Any) -> None:
                unit = units[index]
                if self._settings is not None:
                    cache = store_cache(cache, self._settings)
                rows, steps = groups[index].values.shape
                model = self._tables[unit.table].model
                layout = self._series_layout(model, rows)
                patches = steps // self.series_properties(model).patch
                self._store(unit.key, unit.slots, layout, patches, cache)
                self._fixed[unit.key] = fixed_leaves(layout, cache)
                self.stats.encoded += 1

            self._run_items(
                "series_encode",
                ("series_encode", model, signature),
                compiled,
                (entry.parameters,),
                items,
                estimate,
                indices,
                keep,
            )

    def _extend_series(
        self,
        units: Sequence[SeriesUnit],
        groups: dict[int, SeriesGroup],
        indices: list[int],
    ) -> None:
        """Grows each extended unit's kept encoding by the grid's new patches, read from the slots that hold it, alike ones in one call.

        The grown encoding packs into the unit's own slots with as many
        patches as the source held plus the new ones; the source stays as it
        was. The prefix counts as reused, the unit as extended.
        """
        alike: dict[tuple, list[tuple[int, Any, CacheKey]]] = defaultdict(list)
        for index in indices:
            unit = units[index]
            model = self._tables[unit.table].model
            assert unit.source is not None
            source = self._contexts[unit.source]
            group = groups[index]
            grown = unit.grown
            new = SeriesGroup(
                group.values[:, -grown:],
                group.observed[:, -grown:],
                group.roles,
                None,
                None,
                group.num_rows,
            )
            grown_patches = grown // self.series_properties(model).patch
            held = self._patch_bucket(model, source.rows)
            stored_at = self._patch_bucket(model, source.rows + grown_patches)
            alike[
                (
                    model,
                    source.layout,
                    held,
                    grown_patches,
                    stored_at,
                    _signature((new,)),
                )
            ].append((index, new, unit.source))
        for (
            model,
            layout,
            held,
            grown_patches,
            stored_at,
            _,
        ), members in alike.items():
            entry = self._series(model)
            stored = False

            def build(
                entry: _SeriesModel = entry,
                stored: bool = stored,
                layout: CacheLayout = layout,
                stored_at: int = stored_at,
            ) -> Any:
                return series_extend(entry.functions, stored, layout, stored_at)

            compiled = self.programs.compiled(
                ("series_extend", model, layout, stored_at), build
            )
            # Every source's bucket is read; its real patches are the count.
            positions = np.arange(held, dtype=np.int32)
            blocks = np.stack(
                [
                    np.asarray(self._contexts[source].slots, dtype=np.int32).reshape(
                        -1, layout.slots_per_row
                    )
                    for _, _, source in members
                ]
            )
            device_source, places = self._device.source(blocks)
            items = [
                (
                    places[at],
                    positions,
                    np.int32(self._contexts[source].rows),
                    self._fixed[source],
                    new,
                )
                for at, (_, new, source) in enumerate(members)
            ]

            def estimate(
                layout: CacheLayout = layout,
                patches: int = held + stored_at,
                items: list[tuple] = items,
            ) -> int:
                return (
                    2 * patches * layout.row_bytes
                    + tree_bytes(items[0][3])
                    + tree_bytes(items[0][4])
                )

            def keep(
                index: int,
                cache: Any,
                layout: CacheLayout = layout,
                grown_patches: int = grown_patches,
            ) -> None:
                unit = units[index]
                assert unit.source is not None
                if self._settings is not None:
                    cache = store_cache(cache, self._settings)
                patches = self._contexts[unit.source].rows + grown_patches
                self._store(unit.key, unit.slots, layout, patches, cache)
                self._fixed[unit.key] = fixed_leaves(layout, cache)
                self.stats.extended += 1
                self.stats.reused += 1

            self._run_items(
                "series_extend",
                ("series_extend", model, layout, held, stored_at, grown_patches),
                compiled,
                (entry.parameters, device_source),
                items,
                estimate,
                [index for index, _, _ in members],
                keep,
            )

    def _patch_bucket(self, model: int, patches: int) -> int:
        """The patches an entry of `patches` real ones is stored at, as the node sizes it."""
        return patch_bucket(patches, self.series_properties(model), self._minimum)

    def _decode_series(
        self,
        units: Sequence[SeriesUnit],
        cached: dict[tuple[int, int], list[int]],
        keep: Callable[[tuple[int, int], Any], None],
    ) -> None:
        """Decodes each unit over each padded horizon from its slots, alike ones in one call."""
        alike: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for unit, horizon in cached:
            model = self._tables[units[unit].table].model
            context = self._contexts[units[unit].key]
            bucket = self._patch_bucket(model, context.rows)
            alike[(model, context.layout, bucket, horizon)].append((unit, horizon))
        for (model, layout, patches, horizon), items_keys in alike.items():
            entry = self._series(model)
            # Forecasters read caches as `encode` returned them, so stored
            # dtypes are loaded back first.
            stored = False

            def build(
                entry: _SeriesModel = entry,
                stored: bool = stored,
                layout: CacheLayout = layout,
                horizon: int = horizon,
            ) -> Any:
                return series_decode(entry.functions, stored, layout, horizon)

            compiled = self.programs.compiled(
                ("series_decode", model, layout, horizon), build
            )
            # Every entry's bucket is read; its real patches are the count,
            # so one program serves every count in the bucket.
            positions = np.arange(patches, dtype=np.int32)
            keys = [units[unit].key for unit, _ in items_keys]
            blocks = np.stack(
                [
                    np.asarray(self._contexts[key].slots, dtype=np.int32).reshape(
                        -1, layout.slots_per_row
                    )
                    for key in keys
                ]
            )
            source, places = self._device.source(blocks)
            items = [
                (
                    places[at],
                    positions,
                    np.int32(self._contexts[key].rows),
                    self._fixed[key],
                )
                for at, key in enumerate(keys)
            ]

            def estimate(
                layout: CacheLayout = layout,
                patches: int = patches,
                fixed: Any = self._fixed[keys[0]],
            ) -> int:
                return 2 * patches * layout.row_bytes + tree_bytes(fixed)

            self._run_items(
                "series_decode",
                ("series_decode", model, layout, patches, horizon),
                compiled,
                (entry.parameters, source),
                items,
                estimate,
                items_keys,
                keep,
            )


def _grid_of(unit: SeriesUnit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A unit's grid as host arrays, from the bindings' or a frame's form."""
    grid: Any = getattr(unit, "grid", unit)
    return (
        np.asarray(grid.values, dtype=np.float32),
        np.asarray(grid.observed, dtype=bool),
        np.asarray(grid.roles, dtype=np.int32),
    )


def _future_of(forecast: Forecast) -> tuple[np.ndarray, np.ndarray]:
    """A forecast's known future as host arrays."""
    if isinstance(forecast, protocol.Forecast):
        return forecast.future, forecast.future_observed
    return (
        np.asarray(forecast.future.values, dtype=np.float32),
        np.asarray(forecast.future.observed, dtype=bool),
    )


def check_summary(properties: TabularProperties, summary: str) -> None:
    """Refuses a column-summary scope a model cannot serve.

    Raises:
        ValueError: If `summary` is neither `table` nor `context`, or is
            `context` for a model whose state pools over no context rows.
    """
    if summary not in ("table", "context"):
        raise ValueError(f"summary is table or context, not {summary!r}")
    if summary == "context" and not properties.column_summary:
        raise ValueError(
            "the model has no column summary to fit per context; use summary='table'"
        )


def _stacked_shape(value: Any, count: int, stacked: bool = False) -> Any:
    """The shapes and types of `count` copies of `value` stacked, as a call stacks items.

    With `stacked`, `value` is a tree of leaves, each stacked alone.
    """
    if stacked:
        return jax.tree.map(lambda leaf: _stacked_shape(leaf, count), value)
    return jax.ShapeDtypeStruct(
        (count, *np.shape(value)),
        jnp.result_type(value),
        weak_type=bool(getattr(value, "weak_type", False)),
    )


def _host_padded(values: Any, size: int) -> np.ndarray:
    """Rows padded with zeros to `size` on the host, so no shape compiles a pad."""
    values = np.asarray(values)
    padded = np.zeros((size, *values.shape[1:]), dtype=values.dtype)
    padded[: len(values)] = values
    return padded


def _stacked(items: Sequence[Any]) -> Any:
    """Items of one tree structure stacked leaf by leaf on a new first axis.

    Host arrays stack on the host, so they reach the device in one copy with
    their call rather than one dispatch each. The other leaves stack on the
    device in one program for them all, compiled once per number of items,
    where a stack of each leaf alone would compile once per leaf.
    """
    flat, structures = zip(*(jax.tree.flatten(item) for item in items), strict=True)
    columns = list(zip(*flat, strict=True))
    on_host = [
        all(
            isinstance(value, np.ndarray)
            and jax.dtypes.canonicalize_dtype(value.dtype) == value.dtype
            for value in column
        )
        for column in columns
    ]
    leaves: list[Any] = [
        np.stack(column) if host else None
        for column, host in zip(columns, on_host, strict=True)
    ]
    on_device = [index for index, host in enumerate(on_host) if not host]
    if on_device:
        stacked = _stacked_on_device([columns[index] for index in on_device])
        for index, values in zip(on_device, stacked, strict=True):
            leaves[index] = values
    return jax.tree.unflatten(structures[0], leaves)


def _stacked_on_device(columns: list[tuple]) -> list[jax.Array]:
    """Each column's device values stacked on a new first axis.

    Past `STACK_CHUNK` items the columns stack a chunk at a time and one
    program joins the chunks, so no count compiles a stack of thousands.
    """
    count = len(columns[0])
    if count <= STACK_CHUNK:
        return _stack_columns(columns)
    chunks = [
        _stack_columns([column[start : start + STACK_CHUNK] for column in columns])
        for start in range(0, count, STACK_CHUNK)
    ]
    return _join_columns([list(parts) for parts in zip(*chunks, strict=True)])


@jax.jit
def _stack_columns(columns: list[tuple]) -> list[jax.Array]:
    """Each column's values stacked on a new first axis, in one dispatch."""
    return [jnp.stack(column) for column in columns]


@jax.jit
def _join_columns(columns: list[list[jax.Array]]) -> list[jax.Array]:
    """Each column's stacked chunks joined on their first axis, in one dispatch."""
    return [jnp.concatenate(parts) for parts in columns]


@jax.jit
def _item(outputs: Any, at: jax.Array) -> Any:
    """One item of a call's stacked outputs, in one dispatch rather than one per leaf."""
    return jax.tree.map(lambda value: value[at], outputs)


@functools.partial(jax.jit, donate_argnums=0)
def _only_item(outputs: Any) -> Any:
    """The item of a call's outputs stacked over one, taking over their buffers."""
    return jax.tree.map(lambda value: value[0], outputs)


def _manifests(bucket_minimum: int, slot_bytes: int) -> Path:
    """Where workers record the kinds of call their models compile for, beside JAX's compile cache.

    One directory per version of the code workers trace and of JAX, device
    and padding, since programs recorded under another may not compile, or
    compile alike.
    """
    device = jax.devices()[0]
    identity = "|".join(
        [
            _traced(),
            jax.__version__,
            device.platform,
            device.device_kind,
            *([_cpu_features()] if device.platform == "cpu" else []),
            str(bucket_minimum),
            str(slot_bytes),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return cache_root() / "programs" / digest


def _cpu_features() -> str:
    """The instruction sets XLA may compile this machine's CPU programs with, which a program shipped from another must match.

    The set `portable_cpu_programs` pinned, or else every flag the CPU lists.
    """
    pinned = re.search(r"xla_cpu_max_isa=(\w+)", os.environ.get("XLA_FLAGS", ""))
    if pinned:
        return f"{platform.machine()} {pinned[1]}"
    return " ".join([platform.machine(), *sorted(cpu_flags())])


@functools.cache
def _traced() -> str:
    """A digest of the code workers trace and measure, in place of this package's version.

    A release that leaves this code alone compiles every program as the one
    before did, so it keeps that release's recorded programs and bundles.
    """
    root = Path(__file__).parent
    files = [root / "worker.py"]
    for package in ("models", "nn", "profiling", "runtime"):
        files += (root / package).rglob("*.py")
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(f"{path.relative_to(root).as_posix()}\0".encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _tables(step: protocol.Step) -> list[int]:
    """The tables a step does anything for, in order of first appearance."""
    tables = [fit.table for fit in step.fits]
    tables += [page.table for page in step.pages]
    tables += [unit.table for unit in step.units]
    tables += [batch.table for batch in step.queries]
    tables += [unit.table for unit in step.series_units]
    return list(dict.fromkeys(tables))


def _part(step: protocol.Step, table: int) -> protocol.Step:
    """The part of a step for one table, with its indices renumbered and no copies."""
    pages = [index for index, page in enumerate(step.pages) if page.table == table]
    units = [index for index, unit in enumerate(step.units) if unit.table == table]
    series = [
        index for index, unit in enumerate(step.series_units) if unit.table == table
    ]
    page_at = {old: new for new, old in enumerate(pages)}
    unit_at = {old: new for new, old in enumerate(units)}
    series_at = {old: new for new, old in enumerate(series)}
    return protocol.Step(
        movements=[],
        fits=[
            dataclasses.replace(fit, pages=[page_at[index] for index in fit.pages])
            for fit in step.fits
            if fit.table == table
        ],
        pages=[step.pages[index] for index in pages],
        units=[step.units[index] for index in units],
        queries=[
            dataclasses.replace(batch, units=[unit_at[unit] for unit in batch.units])
            for batch in step.queries
            if batch.table == table
        ],
        series_units=[step.series_units[index] for index in series],
        forecasts=[
            dataclasses.replace(forecast, unit=series_at[forecast.unit])
            for forecast in step.forecasts
            if forecast.table == table
        ],
    )
