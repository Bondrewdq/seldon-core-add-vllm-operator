# vLLM Adapter

The vLLM adapter presents a Seldon REST MODEL server interface to the Seldon executor and forwards non-streaming requests to a vLLM OpenAI-compatible backend.

Request path:

```text
Seldon executor -> POST /predict -> vllm-adapter -> POST /v1/chat/completions -> vLLM
```

Supported endpoints:

- `POST /predict`
- `GET /live`
- `GET /ready`
- `GET /health`
- `GET /health/status`
- `GET /metrics`

Configuration is provided with environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `ADAPTER_HTTP_PORT` | `9000` | HTTP port exposed by the adapter. |
| `VLLM_BASE_URL` | `http://localhost:8000` | Base URL for the vLLM OpenAI-compatible server. |
| `VLLM_MODEL` | required | Model name sent to vLLM. |
| `VLLM_API_KIND` | `chat` | `chat` uses `/v1/chat/completions`; `completions` uses `/v1/completions`. |
| `VLLM_API_KEY` | empty | Optional bearer token for the backend. |
| `REQUEST_TIMEOUT_MS` | `60000` | Downstream request timeout. |
| `DEFAULT_MAX_TOKENS` | `256` | Default `max_tokens` when absent from the request. |
| `DEFAULT_TEMPERATURE` | `0.7` | Default `temperature` when absent from the request. |
| `METRICS_PATH` | `/metrics` | Prometheus metrics endpoint. |

The first implementation intentionally does not support OpenAI streaming responses.
