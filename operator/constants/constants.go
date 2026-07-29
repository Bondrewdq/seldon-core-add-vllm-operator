package constants

const (
	PU_PARAMETER_ENVVAR    = "PREDICTIVE_UNIT_PARAMETERS"
	TFServingContainerName = "tfserving"

	GRPCRegExMatchAmbassador = "/(seldon.protos.*|tensorflow.serving.*|inference.*)/.*"
	GRPCRegExMatchIstio      = ".*tensorflow.*|.*seldon.protos.*|.*inference.*"

	PrePackedServerTensorflow = "TENSORFLOW_SERVER"
	PrePackedServerSklearn    = "SKLEARN_SERVER"
	PrePackedServerTriton     = "TRITON_SERVER"
	PrePackedMlflow           = "MLFLOW_SERVER"
	PrePackedServerVllm       = "VLLM_SERVER"

	TfServingGrpcPort    = 2000
	TfServingRestPort    = 2001
	TfServingArgPort     = "--port="
	TfServingArgRestPort = "--rest_api_port="

	FirstHttpPortNumber   = int32(9000)
	FirstGrpcPortNumber   = int32(9500)
	DNSLocalHost          = "localhost"
	DNSClusterLocalSuffix = ".svc.cluster.local"
	GrpcPortName          = "grpc"
	HttpPortName          = "http"

	TritonDefaultGrpcPort      = 2001
	TritonDefaultHttpPort      = 2000
	TritonArgGrpcPort          = "--grpc-port="
	TritonArgHttpPort          = "--http-port="
	TritonArgModelRepository   = "--model-repository="
	TritonArgModelControlMode  = "--model-control-mode="
	TritonArgLoadModel         = "--load-model="
	TritonArgStrictModelConfig = "--strict-model-config="

	KFServingProbeLivePath  = "/v2/health/live"
	KFServingProbeReadyPath = "/v2/health/ready"

	MLServerDefaultGrpcPort = int32(2001)
	MLServerDefaultHttpPort = int32(2000)
)

const (
	VLLMContainerName               = "vllm"
	VLLMHTTPPortName                = "vllm-http"
	VLLMRuntimeLabelKey             = "seldon.io/runtime"
	VLLMRuntimeLabelValue           = "vllm"
	VLLMModelVolumeName             = "vllm-model"
	VLLMSharedMemoryVolumeName      = "vllm-shm"
	VLLMDefaultImage                = "vllm/vllm-openai:v0.8.5"
	VLLMDefaultHTTPPort             = int32(8081)
	VLLMDefaultRuntimeClassName     = "nvidia"
	VLLMDefaultServedModelName      = "qwen-0.5b"
	VLLMDefaultMaxModelLen          = "1024"
	VLLMDefaultMaxModelLenValue     = int32(1024)
	VLLMDefaultGPUMemoryUtilization = "0.65"
	VLLMDefaultGPUMemoryPercent     = int32(65)
	VLLMDefaultMaxNumSeqs           = "1"
	VLLMDefaultMaxNumSeqsValue      = int32(1)
	VLLMDefaultGPUResourceName      = "nvidia.com/gpu"
	VLLMDefaultGPUCount             = "1"
	VLLMDefaultGPUCountValue        = int32(1)
	VLLMDefaultEnforceEager         = true
	VLLMDefaultSharedMemorySize     = "2Gi"
)

// Metrics-related constants
const (
	FirstMetricsPortNumber = int32(6000)
	DefaultMetricsPortName = "metrics"
)

const (
	ControllerName = "seldon-controller-manager"
)

// Event messages
const (
	EventsCreateAmbassadorMapping = "CreateAmbassadorMapping"
	EventsUpdateAmbassadorMapping = "UpdateAmbassadorMapping"
	EventsDeleteAmbassadorMapping = "DeleteAmbassadorMapping"
	EventsCreateVirtualService    = "CreateVirtualService"
	EventsUpdateVirtualService    = "UpdateVirtualService"
	EventsDeleteVirtualService    = "DeleteVirtualService"
	EventsCreateDestinationRule   = "CreateDestinationRule"
	EventsUpdateDestinationRule   = "UpdateDestinationRule"
	EventsCreateService           = "CreateService"
	EventsUpdateService           = "UpdateService"
	EventsDeleteService           = "DeleteService"
	EventsCreateHPA               = "CreateHPA"
	EventsUpdateHPA               = "UpdateHPA"
	EventsDeleteHPA               = "DeleteHPA"
	EventsCreateScaledObject      = "CreateScaledObject"
	EventsUpdateScaledObject      = "UpdateScaledObject"
	EventsDeleteScaledObject      = "DeleteScaledObject"
	EventsCreatePDB               = "CreatePDB"
	EventsUpdatePDB               = "UpdatePDB"
	EventsDeletePDB               = "DeletePDB"
	EventsCreateDeployment        = "CreateDeployment"
	EventsUpdateDeployment        = "UpdateDeployment"
	EventsDeleteDeployment        = "DeleteDeployment"
	EventsInternalError           = "InternalError"
	EventsUpdated                 = "Updated"
	EventsUpdateFailed            = "UpdateFailed"
)

// Explainers
const (
	ExplainerPathSuffix = "-explainer"
	ExplainerNameSuffix = "-explainer"
)

// Default resources
const (
	DefaultExecutorCpuRequest              = "0.5"
	DefaultExecutorCpuLimit                = "0.5"
	DefaultExecutorMemoryRequest           = "512Mi"
	DefaultExecutorMemoryLimit             = "512Mi"
	DefaultEngineCpuRequest                = "0.5"
	DefaultEngineCpuLimit                  = "0.5"
	DefaultEngineMemoryRequest             = "512Mi"
	DefaultEngineMemoryLimit               = "512Mi"
	DefaultExecutorReqLoggerWorkQueueSize  = "10000"
	DefaultExecutorReqLoggerWriteTimeoutMs = "2000"
)
