package main

import (
	"context"
	"io"
	"log"
	"net"
	"net/http"
	"strconv"
	"testing"
	"time"
)

func TestRunStartsAndStopsApplicationAndMetricsServers(t *testing.T) {
	cfg := Config{
		AdapterHTTPPort:    freeTCPPort(t),
		MetricsHTTPPort:    freeTCPPort(t),
		VLLMBaseURL:        "http://127.0.0.1:1",
		VLLMModel:          "qwen-test",
		VLLMAPIKind:        apiKindChat,
		RequestTimeout:     time.Second,
		DefaultMaxTokens:   64,
		DefaultTemperature: 0.2,
		MetricsPath:        "/prometheus",
		Namespace:          "test-ns",
		SeldonDeployment:   "test-sdep",
		Predictor:          "default",
		Runtime:            "vllm",
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- run(ctx, cfg, log.New(io.Discard, "", 0))
	}()

	waitForHTTPStatus(t, cfg.AdapterHTTPPort, "/live", http.StatusOK)
	waitForHTTPStatus(t, cfg.MetricsHTTPPort, cfg.MetricsPath, http.StatusOK)

	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned an error during graceful shutdown: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("servers did not stop after context cancellation")
	}
}

func freeTCPPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("allocate TCP port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("close TCP listener: %v", err)
	}
	return port
}

func waitForHTTPStatus(t *testing.T, port int, path string, want int) {
	t.Helper()
	url := "http://127.0.0.1:" + strconv.Itoa(port) + path
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == want {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("endpoint %s did not return status %d", url, want)
}
