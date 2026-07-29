package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPredictStrDataBuildsChatCompletionRequest(t *testing.T) {
	var got map[string]interface{}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected backend path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("failed to decode backend request: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":    "chatcmpl-test",
			"model": "qwen-test",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Operator keeps desired state.",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     4,
				"completion_tokens": 5,
				"total_tokens":      9,
			},
		})
	}))
	defer backend.Close()

	server := newTestServer(backend.URL)
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(`{"strData":"Explain Kubernetes Operator"}`))
	rr := httptest.NewRecorder()

	server.routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if got["model"] != "qwen-test" {
		t.Fatalf("expected model qwen-test, got %#v", got["model"])
	}
	messages := got["messages"].([]interface{})
	firstMessage := messages[0].(map[string]interface{})
	if firstMessage["role"] != "user" || firstMessage["content"] != "Explain Kubernetes Operator" {
		t.Fatalf("unexpected messages payload: %#v", got["messages"])
	}
	if got["stream"] != false {
		t.Fatalf("expected stream=false, got %#v", got["stream"])
	}

	var seldonResp SeldonResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &seldonResp); err != nil {
		t.Fatalf("failed to decode adapter response: %v", err)
	}
	if seldonResp.JsonData["text"] != "Operator keeps desired state." {
		t.Fatalf("unexpected response text: %#v", seldonResp.JsonData["text"])
	}
	if seldonResp.Meta["runtime"] != "vllm" {
		t.Fatalf("unexpected meta: %#v", seldonResp.Meta)
	}
}

func TestPredictJSONDataMessagesPreservesRequestOptions(t *testing.T) {
	var got map[string]interface{}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatalf("failed to decode backend request: %v", err)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"model": "custom-model",
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "ok",
					},
				},
			},
		})
	}))
	defer backend.Close()

	server := newTestServer(backend.URL)
	body := `{
		"jsonData": {
			"model": "custom-model",
			"messages": [
				{"role": "system", "content": "You are concise."},
				{"role": "user", "content": "Say ok"}
			],
			"temperature": 0.2,
			"max_tokens": 32
		}
	}`
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(body))
	rr := httptest.NewRecorder()

	server.routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", rr.Code, rr.Body.String())
	}
	if got["model"] != "custom-model" {
		t.Fatalf("expected custom model to be preserved, got %#v", got["model"])
	}
	if got["temperature"].(float64) != 0.2 {
		t.Fatalf("expected temperature 0.2, got %#v", got["temperature"])
	}
	if got["max_tokens"].(float64) != 32 {
		t.Fatalf("expected max_tokens 32, got %#v", got["max_tokens"])
	}
	if got["stream"] != false {
		t.Fatalf("expected stream=false default, got %#v", got["stream"])
	}
}

func TestPredictMissingPromptReturnsBadRequest(t *testing.T) {
	server := newTestServer("http://127.0.0.1:1")
	req := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(`{"data":{"ndarray":[[1,2,3]]}}`))
	rr := httptest.NewRecorder()

	server.routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp SeldonResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Status == nil || resp.Status.Status != "FAILURE" {
		t.Fatalf("expected failure status, got %#v", resp.Status)
	}
}

func TestReadyChecksVLLMHealth(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("unexpected health path: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	server := newTestServer(backend.URL)
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()

	server.routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected ready status 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestReadyFailsWhenVLLMHealthFails(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer backend.Close()

	server := newTestServer(backend.URL)
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()

	server.routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected ready status 503, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestMetricsEndpointExposesAdapterMetrics(t *testing.T) {
	server := newTestServer("http://127.0.0.1:1")
	predictReq := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(`{"data":{"ndarray":[[1,2,3]]}}`))
	predictRR := httptest.NewRecorder()
	server.routes().ServeHTTP(predictRR, predictReq)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()

	server.routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected metrics status 200, got %d", rr.Code)
	}
	body, _ := io.ReadAll(rr.Body)
	if !strings.Contains(string(body), "llm_adapter_requests_total") {
		t.Fatalf("expected adapter metrics, got: %s", string(body))
	}
}

func TestDedicatedSeldonMetricsEndpointExposesAdapterMetrics(t *testing.T) {
	server := newTestServer("http://127.0.0.1:1")
	server.cfg.MetricsPath = "/prometheus"
	predictReq := httptest.NewRequest(http.MethodPost, "/predict", strings.NewReader(`{"data":{"ndarray":[[1,2,3]]}}`))
	predictRR := httptest.NewRecorder()
	server.routes().ServeHTTP(predictRR, predictReq)

	req := httptest.NewRequest(http.MethodGet, "/prometheus", nil)
	rr := httptest.NewRecorder()
	server.metricsRoutes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected metrics status 200, got %d", rr.Code)
	}
	body, _ := io.ReadAll(rr.Body)
	if !strings.Contains(string(body), "llm_adapter_requests_total") {
		t.Fatalf("expected adapter metrics, got: %s", string(body))
	}
}

func newTestServer(baseURL string) *adapterServer {
	cfg := Config{
		AdapterHTTPPort:    9000,
		MetricsHTTPPort:    6000,
		VLLMBaseURL:        baseURL,
		VLLMModel:          "qwen-test",
		VLLMAPIKind:        apiKindChat,
		RequestTimeout:     time.Second,
		DefaultMaxTokens:   128,
		DefaultTemperature: 0.3,
		MetricsPath:        "/metrics",
		Namespace:          "test-ns",
		SeldonDeployment:   "test-sdep",
		Predictor:          "default",
		Runtime:            "vllm",
	}
	return newAdapterServer(cfg, log.New(io.Discard, "", 0))
}
