package main

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	apiKindChat        = "chat"
	apiKindCompletions = "completions"
)

type Config struct {
	AdapterHTTPPort    int
	MetricsHTTPPort    int
	VLLMBaseURL        string
	VLLMModel          string
	VLLMAPIKind        string
	VLLMAPIKey         string
	RequestTimeout     time.Duration
	DefaultMaxTokens   int
	DefaultTemperature float64
	MetricsPath        string
	LogLevel           string

	Namespace        string
	SeldonDeployment string
	Predictor        string
	Runtime          string
}

func loadConfigFromEnv() (Config, error) {
	cfg := Config{
		AdapterHTTPPort:    getEnvInt("ADAPTER_HTTP_PORT", 9000),
		MetricsHTTPPort:    getEnvInt("PREDICTIVE_UNIT_METRICS_SERVICE_PORT", getEnvInt("METRICS_HTTP_PORT", 6000)),
		VLLMBaseURL:        strings.TrimRight(getEnvString("VLLM_BASE_URL", "http://localhost:8000"), "/"),
		VLLMModel:          getEnvString("VLLM_MODEL", ""),
		VLLMAPIKind:        strings.ToLower(getEnvString("VLLM_API_KIND", apiKindChat)),
		VLLMAPIKey:         getEnvString("VLLM_API_KEY", ""),
		RequestTimeout:     time.Duration(getEnvInt("REQUEST_TIMEOUT_MS", 60000)) * time.Millisecond,
		DefaultMaxTokens:   getEnvInt("DEFAULT_MAX_TOKENS", 256),
		DefaultTemperature: getEnvFloat("DEFAULT_TEMPERATURE", 0.7),
		MetricsPath:        firstNonEmpty(os.Getenv("PREDICTIVE_UNIT_METRICS_ENDPOINT"), os.Getenv("METRICS_PATH"), "/prometheus"),
		LogLevel:           getEnvString("LOG_LEVEL", "info"),
		Namespace:          firstNonEmpty(os.Getenv("POD_NAMESPACE"), os.Getenv("NAMESPACE"), "unknown"),
		SeldonDeployment:   firstNonEmpty(os.Getenv("SELDON_DEPLOYMENT_ID"), "unknown"),
		Predictor:          firstNonEmpty(os.Getenv("PREDICTOR_ID"), "unknown"),
		Runtime:            "vllm",
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	if c.AdapterHTTPPort <= 0 || c.AdapterHTTPPort > 65535 {
		return fmt.Errorf("ADAPTER_HTTP_PORT must be a valid TCP port")
	}
	if c.MetricsHTTPPort <= 0 || c.MetricsHTTPPort > 65535 {
		return fmt.Errorf("PREDICTIVE_UNIT_METRICS_SERVICE_PORT must be a valid TCP port")
	}
	if c.MetricsHTTPPort == c.AdapterHTTPPort {
		return fmt.Errorf("metrics and adapter HTTP ports must be different")
	}
	if c.VLLMBaseURL == "" {
		return fmt.Errorf("VLLM_BASE_URL must not be empty")
	}
	if _, err := url.ParseRequestURI(c.VLLMBaseURL); err != nil {
		return fmt.Errorf("VLLM_BASE_URL is invalid: %w", err)
	}
	if c.VLLMModel == "" {
		return fmt.Errorf("VLLM_MODEL must not be empty")
	}
	if c.VLLMAPIKind != apiKindChat && c.VLLMAPIKind != apiKindCompletions {
		return fmt.Errorf("VLLM_API_KIND must be %q or %q", apiKindChat, apiKindCompletions)
	}
	if c.RequestTimeout <= 0 {
		return fmt.Errorf("REQUEST_TIMEOUT_MS must be greater than zero")
	}
	if c.DefaultMaxTokens <= 0 {
		return fmt.Errorf("DEFAULT_MAX_TOKENS must be greater than zero")
	}
	if c.MetricsPath == "" || !strings.HasPrefix(c.MetricsPath, "/") {
		return fmt.Errorf("METRICS_PATH must start with /")
	}
	return nil
}

func getEnvString(key string, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func getEnvFloat(key string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
