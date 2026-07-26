package main

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
)

type adapterMetrics struct {
	registry           *prometheus.Registry
	requestsTotal      *prometheus.CounterVec
	requestErrorsTotal *prometheus.CounterVec
	requestDuration    *prometheus.HistogramVec
	vllmDuration       *prometheus.HistogramVec
	vllmErrorsTotal    *prometheus.CounterVec
	activeRequests     *prometheus.GaugeVec
	tokensTotal        *prometheus.CounterVec
}

func newAdapterMetrics() *adapterMetrics {
	m := &adapterMetrics{
		registry: prometheus.NewRegistry(),
		requestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_adapter_requests_total",
			Help: "Total number of prediction requests handled by the vLLM adapter.",
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime", "status"}),
		requestErrorsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_adapter_request_errors_total",
			Help: "Total number of prediction requests that failed in the vLLM adapter.",
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime", "reason"}),
		requestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llm_adapter_request_duration_seconds",
			Help:    "End-to-end prediction request duration observed by the vLLM adapter.",
			Buckets: prometheus.DefBuckets,
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime", "status"}),
		vllmDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llm_adapter_vllm_request_duration_seconds",
			Help:    "Duration of downstream requests from the adapter to vLLM.",
			Buckets: prometheus.DefBuckets,
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime", "status"}),
		vllmErrorsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_adapter_vllm_errors_total",
			Help: "Total number of downstream vLLM request failures observed by the adapter.",
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime", "status"}),
		activeRequests: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "llm_adapter_active_requests",
			Help: "Number of active prediction requests being handled by the vLLM adapter.",
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime"}),
		tokensTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_adapter_tokens_total",
			Help: "Token counts reported by the OpenAI-compatible vLLM response.",
		}, []string{"namespace", "seldon_deployment", "predictor", "model", "runtime", "type"}),
	}

	m.registry.MustRegister(
		m.requestsTotal,
		m.requestErrorsTotal,
		m.requestDuration,
		m.vllmDuration,
		m.vllmErrorsTotal,
		m.activeRequests,
		m.tokensTotal,
	)
	return m
}

func (m *adapterMetrics) requestLabels(cfg Config, statusCode int) prometheus.Labels {
	return prometheus.Labels{
		"namespace":         cfg.Namespace,
		"seldon_deployment": cfg.SeldonDeployment,
		"predictor":         cfg.Predictor,
		"model":             cfg.VLLMModel,
		"runtime":           cfg.Runtime,
		"status":            strconv.Itoa(statusCode),
	}
}

func (m *adapterMetrics) errorLabels(cfg Config, reason string) prometheus.Labels {
	return prometheus.Labels{
		"namespace":         cfg.Namespace,
		"seldon_deployment": cfg.SeldonDeployment,
		"predictor":         cfg.Predictor,
		"model":             cfg.VLLMModel,
		"runtime":           cfg.Runtime,
		"reason":            reason,
	}
}

func (m *adapterMetrics) activeLabels(cfg Config) prometheus.Labels {
	return prometheus.Labels{
		"namespace":         cfg.Namespace,
		"seldon_deployment": cfg.SeldonDeployment,
		"predictor":         cfg.Predictor,
		"model":             cfg.VLLMModel,
		"runtime":           cfg.Runtime,
	}
}

func (m *adapterMetrics) tokenLabels(cfg Config, tokenType string) prometheus.Labels {
	return prometheus.Labels{
		"namespace":         cfg.Namespace,
		"seldon_deployment": cfg.SeldonDeployment,
		"predictor":         cfg.Predictor,
		"model":             cfg.VLLMModel,
		"runtime":           cfg.Runtime,
		"type":              tokenType,
	}
}
