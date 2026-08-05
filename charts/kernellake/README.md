# kernellake Helm chart

Deploys `kernellake-server` (the Arrow Flight SQL server, see the parent
repo's `docs/ARCHITECTURE.md`) as a plain Kubernetes **Deployment +
Service**. This is explicitly **not** an operator -- no CRDs, no custom
controller. See the parent repo's `docs/ROADMAP.md`, "Explicit non-goals"
section.

## Prerequisites

- A Kubernetes cluster and `helm` 3.x.
- For `backend: gpu` -- a node pool with NVIDIA GPUs and an [NVIDIA device
  plugin](https://github.com/NVIDIA/k8s-device-plugin) already installed on
  the cluster. This chart does not install the device plugin itself; it
  only requests the resulting `nvidia.com/gpu` (or your configured)
  resource on the container.
- For `observability.enabled: true` -- a reachable OTLP collector (gRPC or
  HTTP). If nothing is listening, `kernellake-server` logs clean export
  errors rather than crashing -- see the parent repo's
  `docs/ARCHITECTURE.md`, "OpenTelemetry observability" section.

## Install

```bash
helm install kernellake charts/kernellake                      # CPU backend, defaults
helm install kernellake charts/kernellake --set backend=gpu    # GPU backend
helm install kernellake charts/kernellake \
  --set observability.enabled=true \
  --set observability.otlpEndpoint=otel-collector.observability.svc.cluster.local:4317
```

## Values

| Key | Default | Description |
| --- | --- | --- |
| `image.repository` | `""` (auto: `ghcr.io/hurdad/kernel-lake-cpu` or `-gpu`, picked from `backend` below) | Image repository (built from `docker/Dockerfile`'s `runtime-cpu`/`runtime-gpu` targets); set explicitly to override |
| `image.tag` | `latest` | Image tag |
| `backend` | `cpu` | `cpu` or `gpu` -- mirrors `engine.backend` |
| `service.type` | `ClusterIP` | Kubernetes Service type |
| `service.port` | `31337` | Flight SQL port (matches `ServerSection`'s own default) |
| `gpu.resourceName` | `nvidia.com/gpu` | Extended resource name requested when `backend: gpu` |
| `gpu.count` | `1` | How many GPU resources to request |
| `resources` | `{}` | Standard Kubernetes CPU/memory requests/limits, merged with the automatic GPU request above |
| `nodeSelector` / `tolerations` / `affinity` | `{}` / `[]` / `{}` | Standard Kubernetes scheduling passthrough |
| `config.engine.batchRows` | `1000000` | `engine.batch_rows` |
| `config.engine.resultBatchRows` | `65536` | `engine.result_batch_rows` |
| `config.engine.queryMemoryLimitBytes` | `0` | `engine.query_memory_limit_bytes` -- `0` auto-detects from the GPU's free VRAM |
| `config.storage.localRoot` | `/` | `storage.local_root` |
| `observability.enabled` | `false` | Only takes effect if the image was built with `KERNELLAKE_ENABLE_OTEL=ON` (both `runtime-cpu` and `runtime-gpu` are) |
| `observability.otlpProtocol` | `grpc` | `grpc` or `http` |
| `observability.otlpEndpoint` | `""` | Collector endpoint -- `host:port` for `grpc`, a base URL for `http` |
| `observability.serviceName` | `kernellake-server` | OTel `service.name` resource attribute |
| `observability.useTls` | `false` | Server-CA TLS to the collector (gRPC only) |

This is a **thin** surface over `EngineConfig`
(`include/kernellake/common/config.hpp` in the parent repo) -- any field
not listed above keeps `kernellake-server`'s own compiled-in default. Full
per-signal OTel processor/batch/sampler tuning, mTLS, etc. are not yet
exposed through Helm values; edit `templates/configmap.yaml` directly if
you need them, or use `--set-file` to replace the rendered `kernellake.yaml`.

## Connecting

See `helm install`'s own `NOTES.txt` output after installing -- it prints
the in-cluster DNS name, a `kubectl port-forward` command, and a Python
ADBC Flight SQL client snippet.
