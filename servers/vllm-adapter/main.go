package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	logger := log.New(os.Stdout, "vllm-adapter ", log.LstdFlags|log.LUTC)

	cfg, err := loadConfigFromEnv()
	if err != nil {
		logger.Fatalf("invalid_config error=%q", err)
	}

	server := newAdapterServer(cfg, logger)
	addr := fmt.Sprintf(":%d", cfg.AdapterHTTPPort)
	logger.Printf("starting address=%q vllm_base_url=%q model=%q api_kind=%q metrics_path=%q", addr, cfg.VLLMBaseURL, cfg.VLLMModel, cfg.VLLMAPIKind, cfg.MetricsPath)

	if err := http.ListenAndServe(addr, server.routes()); err != nil {
		logger.Fatalf("server_stopped error=%q", err)
	}
}
