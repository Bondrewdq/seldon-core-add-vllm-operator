# Seldon vLLM observability

This package installs a small observability stack for the VLLM_SERVER runtime:

- kube-prometheus-stack 87.21.0 for Prometheus, Grafana, kube-state-metrics, and kubelet/cAdvisor discovery
- NVIDIA DCGM Exporter 3.3.1 for GPU utilization, framebuffer memory, and temperature
- a PodMonitor for the Seldon adapter, executor, and vLLM endpoints
- two availability/error alerts and one focused Grafana dashboard

The package is separate from the legacy manifests in the parent `monitoring` directory.

## Prerequisites

- the `prometheus-community` Helm repository
- the `nvidia-dcgm` Helm repository at `https://nvidia.github.io/dcgm-exporter/helm-charts`
- an NVIDIA `RuntimeClass` named `nvidia`
- VLLM_SERVER Pods labeled `seldon.io/runtime=vllm`
- the adapter listening on port 6000 at `/prometheus`

## Install

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add nvidia-dcgm https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update

make install
make status
```

The installation intentionally keeps only two days of Prometheus data, disables unrelated control-plane scraping and default dashboards, and does not install Alertmanager.

## Verify targets and queries

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 19090:9090
```

Open `http://127.0.0.1:19090/targets` and verify the Seldon VLLM PodMonitor, DCGM Exporter, kube-state-metrics, and kubelet targets are UP. Core queries include:

```promql
sum(rate(llm_adapter_requests_total[5m]))
histogram_quantile(0.95, sum by (le) (rate(llm_adapter_request_duration_seconds_bucket[5m])))
vllm:num_requests_waiting
histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
sum(rate(vllm:generation_tokens_total[5m]))
vllm:gpu_cache_usage_perc
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_GPU_TEMP
```

Start Grafana with:

```bash
kubectl -n monitoring get secret kube-prometheus-stack-grafana \
  -o jsonpath='{.data.admin-password}' | base64 -d
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 13000:80
```

Open `http://127.0.0.1:13000`, sign in as `admin`, and select the **Seldon vLLM Overview** dashboard. Request-derived panels need real inference traffic and at least two Prometheus scrape samples. DCGM temperature may be absent on WSL2 or consumer GPUs; that is an environment limitation rather than a failed Seldon integration.

## Uninstall

```bash
make uninstall
```

The `monitoring` namespace is retained so uninstalling this package does not remove unrelated resources.
