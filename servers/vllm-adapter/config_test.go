package main

import (
	"strings"
	"testing"
	"time"
)

func TestLoadConfigUsesSeldonMetricsContract(t *testing.T) {
	t.Setenv("VLLM_MODEL", "qwen-test")
	t.Setenv("METRICS_HTTP_PORT", "6200")
	t.Setenv("METRICS_PATH", "/metrics-fallback")
	t.Setenv("PREDICTIVE_UNIT_METRICS_SERVICE_PORT", "6100")
	t.Setenv("PREDICTIVE_UNIT_METRICS_ENDPOINT", "/prometheus")

	cfg, err := loadConfigFromEnv()
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.MetricsHTTPPort != 6100 {
		t.Fatalf("expected Seldon metrics port 6100, got %d", cfg.MetricsHTTPPort)
	}
	if cfg.MetricsPath != "/prometheus" {
		t.Fatalf("expected Seldon metrics path, got %q", cfg.MetricsPath)
	}
}

func TestConfigRejectsMetricsPortConflict(t *testing.T) {
	cfg := Config{
		AdapterHTTPPort:    9000,
		MetricsHTTPPort:    9000,
		VLLMBaseURL:        "http://127.0.0.1:8081",
		VLLMModel:          "qwen-test",
		VLLMAPIKind:        apiKindChat,
		RequestTimeout:     time.Second,
		DefaultMaxTokens:   64,
		DefaultTemperature: 0.2,
		MetricsPath:        "/prometheus",
	}

	err := cfg.Validate()
	if err == nil || !strings.Contains(err.Error(), "must be different") {
		t.Fatalf("expected metrics port conflict, got %v", err)
	}
}
