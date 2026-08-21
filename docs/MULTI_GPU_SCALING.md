# Multi-GPU scaling: plan

Prompted by the GDS investigation in `docs/GPU_OPTIMIZATIONS.md`
surfacing the only two real GDS-validated instance types left standing
(`p5.48xlarge`, 8x H100; `p6-b200.48xlarge`, 8x B200) -- both are
fixed 8-GPU boxes, so using one at all raises the question of whether
KernelLake should do anything with GPUs 2-8 beyond let them sit idle.
This is a planning document only; nothing here is implemented, and
none of it should be started without a separate go-ahead per tier.

Today's architecture is single-GPU by construction, not by an
arbitrary limit that's easy to lift:

- `RmmEnvironment` (`src/memory/rmm_environment.cpp`) installs one RMM
  resource stack as **the process's current CUDA device resource** --
  singular, process-wide, for whatever `EngineConfig::device_id` is.
- `GpuExecutionCoordinator` (`src/server/gpu_execution_coordinator_gpu.cpp`)
  single-flights every query through that one `RmmEnvironment` --
  today's concurrency model is "one query at a time," not "one query
  per GPU."
- `ExecutionContext::cuda_device_id` (`execution_context.hpp`) exists as
  a field but is never set to anything but its `0` default anywhere in
  the tree -- it's a placeholder, not wired-up multi-device support.
- There is exactly one `flight_sql_server.cpp` process, no
  worker/coordinator split, no concept of "node" anywhere in the
  codebase.

Three tiers below, roughly in the order they'd need to be built (each
depends on the previous one's infrastructure):

## Tier 1 (cheap): concurrent queries, one GPU each, single node

**What it is**: turn 8 idle GPUs into 8x concurrent-query throughput.
Each GPU gets its own `RmmEnvironment` instance; `GpuExecutionCoordinator`
picks a free GPU per incoming query instead of mutex-serializing
everything onto device 0. Any single query still runs entirely on one
GPU, capped at that GPU's own memory (80GB H100 / 179GB(ish) B200) --
this tier does not help one query that's too big for one GPU.

**Why it's cheap**: it's mostly the already-scoped opt #2 work
(`GPU_OPTIMIZATIONS.md`'s "drop the `GpuExecutionCoordinator` mutex")
plus a device dimension bolted on, not new architecture. The pre-fix
audit already done for opt #2 (2026-08-17: `ObjectStoreDatasource::device_read()`'s
missing `mr` arg, `literal_to_scalar()`'s missing `stream`/`mr`) applies
identically here -- those gaps must be closed regardless of whether the
result is "N concurrent queries on 1 GPU" or "N concurrent queries
across N GPUs."

**Concrete pieces**:

1. `RmmEnvironment` becomes one-per-device instead of one-per-process:
   either `GpuExecutionCoordinator` owns a `std::vector<std::unique_ptr<RmmEnvironment>>`
   sized to `cudaGetDeviceCount()`, or a `RmmEnvironmentPool` wraps that.
   Each construction call needs `cudaSetDevice(i)` first --
   `set_current_device_resource()` really is one slot per device (confirmed
   from RMM source during the 2026-08-17 mutex investigation), so this
   part is mechanical, not risky.
2. `GpuExecutionCoordinator::execute()` picks a device instead of
   assuming device 0 -- simplest correct policy is round-robin or
   least-recently-used; anything fancier (memory-aware placement) is a
   tier-1.5 refinement, not a blocker.
3. `ExecutionContext::cuda_device_id` gets threaded through for real --
   every `cudaSetDevice()`/stream-creation call in the operator tree
   needs to target the assigned device, not assume "whatever's
   current." This is the part most likely to surface latent
   assume-device-0 bugs; needs the same kind of grep-every-call-site
   audit opt #2 already did for `mr`/`stream`.
4. Per-device metrics: `GpuMemoryMetricsRegistry`
   (`kernellake/memory/gpu_memory_metrics.hpp`) currently reports
   process-wide counters with no device dimension -- needs a device-id
   label added so the existing `kernellake.gpu.memory.*` OTel metrics
   (see the memory-metrics project note) stay meaningful once there's
   more than one device to report on.

**What doesn't change**: no new operator types, no network protocol,
no cross-GPU data movement at all. `HashJoinOperator`,
`ParquetScanOperator`, etc. are untouched -- they just run on whichever
device their `ExecutionContext` says.

**Risk**: low. The main risk is scope creep from "just add a device
loop" into rediscovering all of opt #2's mutex-removal complexity --
budget for that as one combined piece of work, not two.

## Tier 2 (expensive): one query spans multiple GPUs, single node

**What it is**: a single query too big for one GPU's memory (or one
whose join/aggregate would benefit from more aggregate compute) gets
its scan/join/aggregate operators partitioned across the node's GPUs,
with a new exchange step redistributing rows by join/group key between
devices mid-query.

**Why it's expensive**: this needs a genuinely new operator class and
touches the two operators that matter most for correctness
(`HashJoinOperator`, `HashAggregateOperator`), not a config change.

**Real prior art already in the tree, worth building on rather than
starting from scratch**: `HashJoinOperator`
(`src/execution_gpu/hash_join_operator.cpp`) already has a
partition-and-spill path for build sides too large for GPU memory --
`cudf::partitioning` buckets rows by hash of the join key, spills
partitions to Arrow IPC files on disk (`open_partition_reader()`/
`read_partition_batches()`), and processes them incrementally. A
cross-GPU exchange operator is structurally the same idea --
partition rows by hash of the join/group key -- just shipping each
partition to a different device's memory over NVLink instead of to a
local spill file. The partitioning logic transfers almost directly;
what's new is the transport and the operator wiring around it.

**Concrete pieces**:

1. A new `ExchangeOperator` (name TBD) that sits between a scan/join
   and its consumer: partitions its input by a key expression
   (`cudf::partitioning::hash_partition`, already used for the spill
   path above), and for each partition either keeps it local (same
   device) or ships it to the device that owns that partition's hash
   range.
2. Transport for the actual GPU-to-GPU copy: NCCL is the natural
   choice (`ncclSend`/`ncclRecv` point-to-point, not just its
   collective ops) -- it auto-selects NVLink when devices are on the
   same node and connected via NVSwitch (true for all 8 GPUs on both
   `p5.48xlarge` and `p6-b200.48xlarge`), falling back to PCIe P2P
   otherwise. UCX is the alternative if NCCL's collective-library
   shape (designed around training workloads, less around this
   engine's pull-based `next()` operator model) turns out to be an
   awkward fit -- worth a small spike comparing the two APIs against
   `PhysicalOperator::next()`'s pull contract before committing.
3. `RmmEnvironment`/`ExecutionContext::memory_resource` need to accept
   allocations arriving *from* a remote device's exchange, not just
   ones made locally -- today's `limiting_resource_adaptor`/
   `statistics_resource_adaptor` stack (tier 1's per-device split)
   assumes every allocation against a given resource originates on
   that resource's own device.
4. Query planner/physical-plan changes: `physical_planner.cpp` needs
   to decide *when* a query is worth partitioning across devices
   (cost model: estimated build-side size vs. one GPU's
   `query_memory_limit_bytes`) and insert `ExchangeOperator` nodes
   into the physical plan accordingly -- this is a new planning
   decision, not just a new operator that's always used.
5. Partial-aggregate correctness: `HashAggregateOperator`'s
   `max_distinct_keys` cap (the SF100 Q3 fix from 2026-08-09) becomes
   a per-partition, not global, constraint once aggregation is split
   across devices -- needs a real re-check, not an assumption that
   splitting the input makes the cap issue strictly easier.

**Risk**: high. This is the one place a subtly wrong partition
boundary produces silently incorrect query results rather than a
crash -- needs the same discipline as the Q3 predicate-pushdown fix
(exact-row-match validation against DuckDB, not just "it completes").

## Tier 3 (multi-node): multiple instances, each with multiple GPUs

**What it is**: scale-out across `p5.48xlarge`/`p6-b200.48xlarge`
instances, not just across the 8 GPUs within one. Requires everything
tier 2 built, plus:

1. **A coordinator/worker split that doesn't exist today.**
   `flight_sql_server.cpp` is one process handling one client
   connection's worth of query execution directly -- multi-node needs
   a coordinator that plans a query, fragments it across worker nodes
   (each worker owning some subset of GPUs, running today's
   single-node engine internally), and collects results. This is a new
   process topology, not an extension of the existing server binary.
2. **Cross-node transport for the exchange operator.** Tier 2's NCCL
   path over NVLink doesn't span nodes -- cross-node traffic needs
   NCCL over network RDMA (works if EFA is present and NCCL's AWS-OFI
   plugin is installed) or UCX with its own RDMA transport. This is
   the one place the GDS investigation's EFA work is directly
   reusable: whichever instance type is chosen already has EFA
   hardware (both `p5.48xlarge` and `p6-b200.48xlarge` do, per the
   `describe-instance-types` check during the GDS work) -- the EFA
   *software* stack (the `aws-efa-installer` step from the GDS
   testing) is a real prerequisite here too, independent of whether
   GDS itself ever works.
3. **Data locality for scans.** `ParquetScanOperator` reading from S3
   today has no node-affinity concept -- a distributed query needs to
   decide *which worker* reads which files/row-groups (ideally
   whichever worker will also own that partition's join/aggregate
   work, to avoid an extra network hop). This is new planning logic,
   analogous to Spark's own scan-task-to-executor assignment.
4. **Arrow Flight as the inter-node wire protocol, not just the
   client-facing one.** `flight_sql_server.cpp` already speaks Arrow
   Flight to clients -- reusing Flight (not a bespoke RPC protocol)
   for coordinator-to-worker and worker-to-worker result shipping
   keeps the wire format consistent with what the codebase already
   depends on, and Arrow Flight's `DoExchange` RPC is a reasonable fit
   for streaming partitioned `RecordBatch` traffic between workers.
   Worth a design spike specifically comparing Flight's `DoExchange`
   against a raw NCCL/UCX path for the worker-to-worker leg -- Flight
   is simpler and already a dependency, NCCL/UCX+RDMA is faster but a
   new dependency and a new failure-mode surface (partial-write
   recovery, connection teardown under a killed worker, etc.).
5. **Fault handling.** Single-node/single-GPU KernelLake today has no
   notion of "a piece of the query failed but the rest can continue or
   retry" -- a worker node dying mid-query is a new failure mode this
   engine has never had to reason about.

**Risk**: very high, and the least like anything in this codebase
today. This tier is closer to "build a distributed query engine" than
"extend KernelLake" -- worth treating as its own project with its own
scoping, not a natural next step after tier 2.

## Recommendation

Build tier 1 regardless of which instance type (or whether either) is
ever provisioned -- it's cheap, mostly already-scoped work, and pays
off even on hardware already owned (any multi-GPU box, not just the
two GDS-validated ones). Tier 2 only pays for itself if a real
single-query workload too big for one GPU's memory actually shows up
(nothing measured in this project so far has hit that ceiling -- SF1000
Q3 fits in 9.9 GiB post-fix, well under even the smaller GPUs already
tested). Tier 3 is a separate, much larger decision -- don't scope it
until tier 2 is real and there's a concrete reason (data volume, SLA)
that one node's worth of GPUs genuinely isn't enough.

## Open questions, not yet resolved

- Does `p6-b200.48xlarge`'s NVLink topology give all-to-all bandwidth
  identical to `p5.48xlarge`'s, or does the newer NVSwitch generation
  change tier 2's exchange-operator cost model? Not checked.
- NCCL vs. UCX for tier 2's point-to-point transport -- leaning NCCL
  for AWS-ecosystem maturity (the AWS-OFI-NCCL plugin is
  Amazon-maintained specifically for EFA), but no spike has been done
  against this engine's actual `next()`-based pull model.
- Whether tier 3's coordinator should be a new binary or a mode of the
  existing `flight_sql_server.cpp` -- not decided, affects how much of
  `main.cpp`/`flight_sql_server.cpp` gets touched vs. left alone.
