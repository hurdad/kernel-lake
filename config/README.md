# Config files

Two default config files, one per binary -- neither is optional
boilerplate, both are loaded automatically when the matching binary is
run with no `--config` flag:

- **`kernellake-cli.yaml`** -- loaded by the `kernellake` CLI (`query`,
  `explain`, `inspect-parquet`, `benchmark tpch`). Contains only the
  top-level keys `CliConfig` reads: `engine.device_id` (which GPU an
  ad-hoc query runs on) and `benchmark:` (default iteration counts),
  plus every section shared with the server (`memory`, `storage`,
  `iceberg`, `unity_catalog`, `delta`, `logging`, `profiling`,
  `observability`, and the rest of `engine`).
- **`kernellake-server.yaml`** -- loaded by `kernellake-server` (the
  Arrow Flight SQL server). Contains only the top-level keys
  `ServerConfig` reads: `server:` (host/port/TLS/auth),
  `engine.max_concurrent_gpu_queries` and `engine.gpu_device_ids`
  (per-GPU concurrency cap and which GPUs to use), plus the same
  shared sections as the CLI file.

Why two files rather than one shared file both binaries read: a field
that only applies to one binary used to live on one shared config
struct, which meant it was silently ignored by whichever binary didn't
use it (`engine.device_id` specifically, once `kernellake-server`
moved to a per-device GPU model and stopped reading it at all). Splitting
the config *type* itself — `EngineConfig` (shared) plus `CliConfig`/
`ServerConfig` (each binary's own top-level type, see
`include/kernellake/common/config.hpp`) — makes that class of bug
structurally impossible: a binary's config type simply doesn't have a
field it wouldn't use.

Point either binary at a different file with `--config <path>`:

```
kernellake --config /path/to/my-cli-config.yaml query --sql "..."
kernellake-server --config /path/to/my-server-config.yaml
```

Every field is optional -- an omitted key falls back to its compiled-in
default (see `include/kernellake/common/config.hpp` for every field's
own doc comment). Both files here list every field explicitly for
documentation purposes, not because they're all required.

Other config files elsewhere in this repo build on these two rather
than replacing them:

- `benchmarks/local/config/kernellake-server.yaml` -- a real deployment
  example for the local docker-compose stack (MinIO/Prometheus/Grafana/
  Jaeger), mirrors `kernellake-server.yaml`'s shape with stack-specific
  values filled in.
- `charts/kernellake/templates/configmap.yaml` -- the Helm chart renders
  its own `kernellake-server.yaml` from a small values.yaml surface
  (the chart only ever deploys `kernellake-server`, never the CLI).
