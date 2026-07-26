FROM golang:1.17-bullseye AS builder

WORKDIR /workspace
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o vllm-adapter .

FROM gcr.io/distroless/static-debian11:nonroot
WORKDIR /
COPY --from=builder /workspace/vllm-adapter /vllm-adapter
USER nonroot:nonroot
ENTRYPOINT ["/vllm-adapter"]
