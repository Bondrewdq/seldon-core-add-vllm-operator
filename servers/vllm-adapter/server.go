package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type adapterServer struct {
	cfg     Config
	client  *http.Client
	metrics *adapterMetrics
	logger  *log.Logger
}

func newAdapterServer(cfg Config, logger *log.Logger) *adapterServer {
	return &adapterServer{
		cfg: cfg,
		client: &http.Client{
			Timeout: cfg.RequestTimeout,
		},
		metrics: newAdapterMetrics(),
		logger:  logger,
	}
}

func (s *adapterServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/predict", s.handlePredict)
	mux.HandleFunc("/live", s.handleLive)
	mux.HandleFunc("/ready", s.handleReady)
	mux.HandleFunc("/health", s.handleReady)
	mux.HandleFunc("/health/status", s.handleReady)
	metricsHandler := promhttp.HandlerFor(s.metrics.registry, promhttp.HandlerOpts{})
	mux.Handle("/metrics", metricsHandler)
	if s.cfg.MetricsPath != "/metrics" {
		mux.Handle(s.cfg.MetricsPath, metricsHandler)
	}
	return mux
}

func (s *adapterServer) metricsRoutes() http.Handler {
	mux := http.NewServeMux()
	mux.Handle(s.cfg.MetricsPath, promhttp.HandlerFor(s.metrics.registry, promhttp.HandlerOpts{}))
	return mux
}

func (s *adapterServer) handleLive(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (s *adapterServer) handleReady(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
	defer cancel()

	if err := s.checkVLLMReady(ctx); err != nil {
		writeJSON(w, http.StatusServiceUnavailable, SeldonResponse{
			Status: failureStatus(http.StatusServiceUnavailable, err.Error()),
			Meta:   s.responseMeta(),
		})
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (s *adapterServer) handlePredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	statusCode := http.StatusOK
	reason := ""
	labels := s.metrics.activeLabels(s.cfg)
	s.metrics.activeRequests.With(labels).Inc()
	defer s.metrics.activeRequests.With(labels).Dec()

	defer func() {
		s.metrics.requestsTotal.With(s.metrics.requestLabels(s.cfg, statusCode)).Inc()
		s.metrics.requestDuration.With(s.metrics.requestLabels(s.cfg, statusCode)).Observe(time.Since(start).Seconds())
		if statusCode >= 400 {
			s.metrics.requestErrorsTotal.With(s.metrics.errorLabels(s.cfg, reason)).Inc()
		}
	}()

	if r.Method != http.MethodPost {
		statusCode = http.StatusMethodNotAllowed
		reason = "method_not_allowed"
		writeJSON(w, statusCode, SeldonResponse{
			Status: failureStatus(statusCode, "method not allowed"),
			Meta:   s.responseMeta(),
		})
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		statusCode = http.StatusBadRequest
		reason = "bad_request"
		writeJSON(w, statusCode, SeldonResponse{
			Status: failureStatus(statusCode, "failed to read request body"),
			Meta:   s.responseMeta(),
		})
		return
	}

	vllmBody, err := s.buildVLLMRequest(body)
	if err != nil {
		statusCode, reason = classifyError(err)
		writeJSON(w, statusCode, SeldonResponse{
			Status: failureStatus(statusCode, err.Error()),
			Meta:   s.responseMeta(),
		})
		return
	}

	vllmStart := time.Now()
	vllmResp, vllmStatus, err := s.callVLLM(r.Context(), vllmBody)
	s.metrics.vllmDuration.With(s.metrics.requestLabels(s.cfg, vllmStatus)).Observe(time.Since(vllmStart).Seconds())
	if err != nil {
		statusCode, reason = classifyError(err)
		s.metrics.vllmErrorsTotal.With(s.metrics.requestLabels(s.cfg, statusCode)).Inc()
		writeJSON(w, statusCode, SeldonResponse{
			Status: failureStatus(statusCode, err.Error()),
			Meta:   s.responseMeta(),
		})
		return
	}

	resp, err := s.buildSeldonResponse(vllmResp)
	if err != nil {
		statusCode = http.StatusBadGateway
		reason = "invalid_vllm_response"
		writeJSON(w, statusCode, SeldonResponse{
			Status: failureStatus(statusCode, err.Error()),
			Meta:   s.responseMeta(),
		})
		return
	}

	if resp.JsonData == nil {
		resp.JsonData = map[string]interface{}{}
	}
	resp.Meta = s.responseMeta()
	writeJSON(w, http.StatusOK, resp)
	s.logger.Printf("request_completed status=%d duration_ms=%d model=%q", http.StatusOK, time.Since(start).Milliseconds(), s.cfg.VLLMModel)
}

func (s *adapterServer) buildVLLMRequest(body []byte) ([]byte, error) {
	var req SeldonRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, &adapterError{statusCode: http.StatusBadRequest, reason: "bad_json", message: "failed to parse Seldon request JSON"}
	}

	if len(req.JsonData) > 0 && string(req.JsonData) != "null" {
		return s.buildFromJSONData(req.JsonData)
	}

	prompt, err := promptFromSeldonRequest(req)
	if err != nil {
		return nil, err
	}

	if s.cfg.VLLMAPIKind == apiKindCompletions {
		return json.Marshal(map[string]interface{}{
			"model":       s.cfg.VLLMModel,
			"prompt":      prompt,
			"max_tokens":  s.cfg.DefaultMaxTokens,
			"temperature": s.cfg.DefaultTemperature,
			"stream":      false,
		})
	}

	return json.Marshal(map[string]interface{}{
		"model": s.cfg.VLLMModel,
		"messages": []ChatMessage{
			{Role: "user", Content: prompt},
		},
		"max_tokens":  s.cfg.DefaultMaxTokens,
		"temperature": s.cfg.DefaultTemperature,
		"stream":      false,
	})
}

func (s *adapterServer) buildFromJSONData(raw json.RawMessage) ([]byte, error) {
	var payload map[string]interface{}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, &adapterError{statusCode: http.StatusBadRequest, reason: "bad_jsondata", message: "jsonData must be a JSON object"}
	}

	if _, ok := payload["model"]; !ok {
		payload["model"] = s.cfg.VLLMModel
	}
	if _, ok := payload["stream"]; !ok {
		payload["stream"] = false
	}
	if _, ok := payload["max_tokens"]; !ok && s.cfg.DefaultMaxTokens > 0 {
		payload["max_tokens"] = s.cfg.DefaultMaxTokens
	}
	if _, ok := payload["temperature"]; !ok {
		payload["temperature"] = s.cfg.DefaultTemperature
	}

	if _, ok := payload["messages"]; ok {
		return json.Marshal(payload)
	}

	if prompt, ok := stringFromValue(payload["prompt"]); ok {
		if s.cfg.VLLMAPIKind == apiKindCompletions {
			payload["prompt"] = prompt
			return json.Marshal(payload)
		}
		delete(payload, "prompt")
		payload["messages"] = []ChatMessage{{Role: "user", Content: prompt}}
		return json.Marshal(payload)
	}

	return nil, &adapterError{statusCode: http.StatusBadRequest, reason: "missing_prompt", message: "jsonData must include messages or prompt"}
}

func (s *adapterServer) callVLLM(ctx context.Context, body []byte) ([]byte, int, error) {
	endpoint := "/v1/chat/completions"
	if s.cfg.VLLMAPIKind == apiKindCompletions {
		endpoint = "/v1/completions"
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.VLLMBaseURL+endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, http.StatusInternalServerError, &adapterError{statusCode: http.StatusInternalServerError, reason: "internal", message: err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	if s.cfg.VLLMAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.cfg.VLLMAPIKey)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, http.StatusGatewayTimeout, &adapterError{statusCode: http.StatusGatewayTimeout, reason: "vllm_timeout", message: "vLLM request timed out"}
		}
		return nil, http.StatusServiceUnavailable, &adapterError{statusCode: http.StatusServiceUnavailable, reason: "vllm_unavailable", message: "vLLM backend unavailable: " + err.Error()}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, http.StatusBadGateway, &adapterError{statusCode: http.StatusBadGateway, reason: "vllm_read_error", message: "failed to read vLLM response"}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, resp.StatusCode, &adapterError{statusCode: http.StatusBadGateway, reason: "vllm_bad_status", message: fmt.Sprintf("vLLM returned status %d: %s", resp.StatusCode, truncate(string(respBody), 512))}
	}
	return respBody, resp.StatusCode, nil
}

func (s *adapterServer) buildSeldonResponse(body []byte) (SeldonResponse, error) {
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return SeldonResponse{}, fmt.Errorf("failed to parse vLLM response JSON")
	}

	var typed openAIResponse
	if err := json.Unmarshal(body, &typed); err != nil {
		return SeldonResponse{}, fmt.Errorf("failed to parse OpenAI-compatible response")
	}

	text := extractText(typed)
	decoded["text"] = text
	if typed.Model != "" {
		decoded["model"] = typed.Model
	}
	if typed.Usage != nil {
		s.observeUsage(*typed.Usage)
	}

	return SeldonResponse{JsonData: decoded}, nil
}

func (s *adapterServer) checkVLLMReady(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.cfg.VLLMBaseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("vLLM health check returned status %d", resp.StatusCode)
	}
	return nil
}

func (s *adapterServer) observeUsage(usage Usage) {
	if usage.PromptTokens > 0 {
		s.metrics.tokensTotal.With(s.metrics.tokenLabels(s.cfg, "prompt")).Add(float64(usage.PromptTokens))
	}
	if usage.CompletionTokens > 0 {
		s.metrics.tokensTotal.With(s.metrics.tokenLabels(s.cfg, "completion")).Add(float64(usage.CompletionTokens))
	}
	if usage.TotalTokens > 0 {
		s.metrics.tokensTotal.With(s.metrics.tokenLabels(s.cfg, "total")).Add(float64(usage.TotalTokens))
	}
}

func promptFromSeldonRequest(req SeldonRequest) (string, error) {
	if strings.TrimSpace(req.StrData) != "" {
		return req.StrData, nil
	}

	if req.Data != nil && len(req.Data.Ndarray) > 0 {
		var value interface{}
		if err := json.Unmarshal(req.Data.Ndarray, &value); err != nil {
			return "", &adapterError{statusCode: http.StatusBadRequest, reason: "bad_ndarray", message: "data.ndarray must be valid JSON"}
		}
		if prompt, ok := firstString(value); ok {
			return prompt, nil
		}
	}

	return "", &adapterError{statusCode: http.StatusBadRequest, reason: "missing_prompt", message: "request must include jsonData.messages, jsonData.prompt, strData, or data.ndarray string"}
}

func firstString(value interface{}) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, strings.TrimSpace(v) != ""
	case []interface{}:
		for _, item := range v {
			if s, ok := firstString(item); ok {
				return s, true
			}
		}
	}
	return "", false
}

func stringFromValue(value interface{}) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, strings.TrimSpace(v) != ""
	default:
		return "", false
	}
}

func extractText(resp openAIResponse) string {
	if len(resp.Choices) == 0 {
		return ""
	}
	if resp.Choices[0].Message != nil {
		return resp.Choices[0].Message.Content
	}
	return resp.Choices[0].Text
}

func (s *adapterServer) responseMeta() map[string]interface{} {
	return map[string]interface{}{
		"runtime":           s.cfg.Runtime,
		"model":             s.cfg.VLLMModel,
		"namespace":         s.cfg.Namespace,
		"seldonDeployment":  s.cfg.SeldonDeployment,
		"predictor":         s.cfg.Predictor,
		"openaiApiKind":     s.cfg.VLLMAPIKind,
		"openaiBackendHost": s.cfg.VLLMBaseURL,
	}
}

func failureStatus(code int, info string) *SeldonStatus {
	return &SeldonStatus{
		Code:   code,
		Info:   info,
		Status: "FAILURE",
	}
}

func classifyError(err error) (int, string) {
	var adapterErr *adapterError
	if errors.As(err, &adapterErr) {
		return adapterErr.statusCode, adapterErr.reason
	}
	return http.StatusInternalServerError, "internal"
}

func writeJSON(w http.ResponseWriter, statusCode int, value interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(value)
}

func truncate(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit]
}
