# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tunable runtime constants for the worker and its runtime."""

# --- Worker ---

# XLA's GPU autotuner times kernels on buffers of its own while it compiles,
# beside what a step holds; a call leaves it a sixteenth of its working memory.
AUTOTUNE_SHARE = 16

# What `Worker.fit_bound` takes a fit to hold per cell, and per byte of its
# rows' representations: built-in models' fits hold at most about a fifth of
# the first, measured to 1,000 columns, and a sixtieth of the second.
FIT_CELL_BYTES = 1 << 16
FIT_ROW_SCALE = 64

# Rows a thread scatters at a time, so a step's few large decodings spread
# over the scatter's threads: at 999 float32 columns, 16 MiB a piece, several
# pieces a thread.
SCATTER_ROWS = 4096
# The most threads a worker scatters a step's answers on, past which the
# scatter's share of a step barely shrinks and workers sharing a host's cores
# would contend for them.
SCATTER_THREADS = 8
# Items a device stack takes in one program. A stack's compile grows faster
# than its items, about 14 s at 2,048 on a CPU, so a larger stack joins
# stacks of this many, which compile in a tenth of that and run as fast.
STACK_CHUNK = 64

# --- Memory measurement ---

# Context sizes the cache is measured at. Tracing does no arithmetic, so any two work.
CACHE_TRACE_SMALL, CACHE_TRACE_LARGE = 8, 16
# Context sizes a decode or an encode is measured at: compiling allocates
# nothing, and at these what grows with the context's rows, such as attention
# scores, dominates.
WORKING_ROWS = (2048, 4096)
# The fewest context rows a fit from those sizes holds for.
MEASURED_ROWS = WORKING_ROWS[0]
# Table widths a model's calls are measured at, once per model. An encode's
# temporaries can grow faster than its cache with width, so a wider table's
# encode is measured at the table's own width.
WORKING_WIDTHS = (4, 16)
# Headroom over the estimate, which XLA's choices at other sizes can exceed.
WORKING_MARGIN = 1.05
# Rows a model's fit is measured at, once per model and table width. Below a
# few thousand rows, what XLA holds does not yet grow with them in a line.
FIT_ROWS = (4096, 16384)
# Headroom over a fit's measure, which XLA's choices between the sizes it was
# measured at, and past them, can exceed.
FIT_MARGIN = 1.1
# Rows a step of a measured fit's column summary reads, so what the fit holds
# per row besides its steps shows.
FIT_STEP_ROWS = 8
# The fewest rows a step of a streamed column summary reads, whatever the call
# budget leaves it: fewer would loop over the rows for little memory saved.
MIN_SUMMARY_ROWS = 64
# The fewest rows a call budget limits a state's fit to: fewer barely describe
# a table, and a call past the budget at that size runs anyway, as any call
# too large for it alone does.
MIN_FIT_ROWS = 4096

# --- Programs ---

# Table shapes, a model, task and width each, a worker keeps compiled programs
# for while memory is short. JAX keeps every program it compiles, about 200 MiB
# of host memory a shape on a GPU, so a worker past this many, short of memory
# by its budget, drops them all; those still used compile again, which with
# many widths costs far more than the memory, so plenty keeps them.
COMPILED_SHAPES = 32
# The share of Linux's memory mappings a process may hold before its worker
# drops its programs whatever their count: one worker compiling many shapes
# maps code for each, and a compile past the cap fails.
MAPPINGS_SHORT = 3 / 4
# Stages whose one call reads a whole table or context, as fitting its state
# or encoding it: seconds at a large one, so a warm start runs them on zeros
# only up to `WARM_RUN_ROWS` rows, and past that only compiles them.
TABLE_STAGES = frozenset({"fit", "state", "encode"})
WARM_RUN_ROWS = 1024
# Page counts a table's embeds compile for once it is fit, this many buckets
# from one page: a write changes one page or a few.
WARMED_PAGES = 4
# Query counts a table's query embeds and decodes compile for once it has
# decoded, this many buckets above the smallest: steps merging many callers'
# requests reach them.
WARMED_QUERIES = 5
# The stacked counts an accelerator's fused calls take, each its own program
# that compiles for tens of seconds: a step's new contexts pad up to one, or
# split across the most, rather than compile for every power of two.
FUSED_STACKS = (1, 4, 16, 64)
# Seconds a worker goes without a step before compiling ahead. A caller sending
# steps back to back would otherwise meet a compile in each of its next steps.
WARM_QUIET = 0.5
# Kinds of call a model's manifest keeps, the latest used last: each table
# shape compiles about a kind per stage, so this keeps `COMPILED_SHAPES`'s.
RECORDED_KINDS = 8 * COMPILED_SHAPES

# --- Paging ---

# Two traced context sizes; the axis that grows between them is the row axis.
PAGE_TRACE_SMALL, PAGE_TRACE_LARGE = 16, 48
# Two traced patch counts of a series; the axis that grows between them is the
# patch axis.
SERIES_TRACE_SMALL, SERIES_TRACE_LARGE = 2, 4
# Slots a move between tiers copies at a time. Each part but the last is this
# many, and the last pads to a bucket, so moves compile a few programs and
# put at most a part on the device beside the pool.
MOVE_SLOTS = 64

# Bytes of a cache packed at a time as it is written to the device pool,
# rounded down to a power of two of blocks.
PACK_BYTES = 1 << 30

# --- Device ---

# The most device memory a local client keeps contexts in by default.
LOCAL_DEVICE_TIER = 4 << 30
# Attention blocks take at most this many positions per step, so large tables
# fit in memory; smaller ones run in one step.
ATTENTION_BLOCK_CHUNK = 1 << 18

# --- Profiling ---

# How often the sampler reads every thread's stack.
SAMPLE_INTERVAL = 0.001
