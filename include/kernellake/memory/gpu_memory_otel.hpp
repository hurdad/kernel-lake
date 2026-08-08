#pragma once

namespace kernellake {

// Registers the kernellake.gpu.memory.* OTel metric instruments (see
// docs/OBSERVABILITY.md) against whatever global MeterProvider
// observability::init() already installed -- an ObservableGauge each for
// current/peak bytes, an ObservableCounter each for allocations/
// deallocations/allocated_total_bytes/allocation_failures, every callback
// reading GpuMemoryMetricsRegistry snapshots (never touching the
// allocation hot path itself). Idempotent (internally std::call_once-
// guarded) and safe to call from multiple threads/multiple
// RmmEnvironment constructions -- only the first call does anything.
//
// Deliberately independent of whether observability is actually enabled or
// this is even a KERNELLAKE_ENABLE_OTEL build: the no-otel build's stub
// implementation is a plain no-op (mirroring kernellake::observability's
// own init()/query_tracing_stub.cpp split), and an OTel build with
// observability.enabled=false still registers instruments against OTel's
// own default no-op MeterProvider -- harmless, since nothing ever reads
// their callbacks in that case either. Called once from RmmEnvironment's
// constructor (src/memory/rmm_environment.cpp), not from
// observability::init() itself, since kernellake_observability has no
// CUDA/RMM dependency and must stay that way (it's built in every preset,
// including CPU-only ones) -- see this pair's own .cpp files' comments.
void register_gpu_memory_otel_instruments();

// Test-only: undoes register_gpu_memory_otel_instruments()'s call_once
// guard and drops the currently-held instrument handles, so a test that
// calls kernellake::observability::init_for_testing() (swapping in a fresh,
// in-memory MeterProvider -- see query_tracing_test_support.hpp) *after*
// something else in the same test binary already triggered real
// registration (binding these instruments to whatever MeterProvider was
// current at that first-ever call, likely OTel's default no-op one) can
// force a fresh registration against the provider it actually wants to
// assert against. Never called from production code -- see
// gpu_memory_metrics_test.cpp for the only real use. No-op in the
// KERNELLAKE_ENABLE_OTEL=OFF stub.
void reset_gpu_memory_otel_instruments_for_testing();

}  // namespace kernellake
