package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

const serverShutdownTimeout = 5 * time.Second

func main() {
	logger := log.New(os.Stdout, "vllm-adapter ", log.LstdFlags|log.LUTC)

	cfg, err := loadConfigFromEnv()
	if err != nil {
		logger.Fatalf("invalid_config error=%q", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx, cfg, logger); err != nil {
		logger.Fatalf("server_stopped error=%q", err)
	}
}

func run(ctx context.Context, cfg Config, logger *log.Logger) error {
	adapter := newAdapterServer(cfg, logger)
	applicationServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.AdapterHTTPPort),
		Handler: adapter.routes(),
	}
	metricsServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.MetricsHTTPPort),
		Handler: adapter.metricsRoutes(),
	}

	logger.Printf("starting application_address=%q metrics_address=%q metrics_path=%q vllm_base_url=%q model=%q api_kind=%q", applicationServer.Addr, metricsServer.Addr, cfg.MetricsPath, cfg.VLLMBaseURL, cfg.VLLMModel, cfg.VLLMAPIKind)
	return runHTTPServers(ctx, applicationServer, metricsServer)
}

func runHTTPServers(ctx context.Context, servers ...*http.Server) error {
	errCh := make(chan error, len(servers))
	for _, server := range servers {
		server := server
		go func() {
			if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				errCh <- fmt.Errorf("listen on %s: %w", server.Addr, err)
			}
		}()
	}

	var runErr error
	select {
	case <-ctx.Done():
	case runErr = <-errCh:
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), serverShutdownTimeout)
	defer cancel()
	for _, server := range servers {
		if err := server.Shutdown(shutdownCtx); err != nil && runErr == nil {
			runErr = fmt.Errorf("shut down %s: %w", server.Addr, err)
		}
	}
	return runErr
}
