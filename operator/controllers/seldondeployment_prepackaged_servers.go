/*
Copyright 2019 The Seldon Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	machinelearningv1 "github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
	"github.com/seldonio/seldon-core/operator/constants"
	"github.com/seldonio/seldon-core/operator/utils"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
)

type PrePackedInitialiser struct {
	clientset kubernetes.Interface
	ctx       context.Context
}

func NewPrePackedInitializer(ctx context.Context, clientset kubernetes.Interface) *PrePackedInitialiser {
	return &PrePackedInitialiser{clientset: clientset, ctx: ctx}
}

func extractEnvSecretRefName(pu *machinelearningv1.PredictiveUnit) string {
	envSecretRefName := ""
	if pu.EnvSecretRefName == "" {
		envSecretRefName = PredictiveUnitDefaultEnvSecretRefName
	} else {
		envSecretRefName = pu.EnvSecretRefName
	}
	return envSecretRefName
}

func createTensorflowServingContainer(mlDepSepc *machinelearningv1.SeldonDeploymentSpec, pu *machinelearningv1.PredictiveUnit, tensorflowProtocol bool) *v1.Container {
	ServerConfig := machinelearningv1.GetPrepackServerConfig(string(*pu.Implementation))

	tfImage := ServerConfig.PrepackImageName(machinelearningv1.ProtocolTensorflow, pu)

	grpcPort := int32(constants.TfServingGrpcPort)
	restPort := int32(constants.TfServingRestPort)
	name := constants.TFServingContainerName
	if tensorflowProtocol {
		grpcPort = pu.Endpoint.GrpcPort
		restPort = pu.Endpoint.HttpPort
		name = pu.Name
	}

	return &v1.Container{
		Name:  name,
		Image: tfImage,
		Args: []string{
			constants.TfServingArgPort + strconv.Itoa(int(grpcPort)),
			constants.TfServingArgRestPort + strconv.Itoa(int(restPort)),
			"--model_name=" + pu.Name,
			"--model_base_path=" + DefaultModelLocalMountPath},
		ImagePullPolicy: v1.PullIfNotPresent,
		Ports: []v1.ContainerPort{
			{
				ContainerPort: grpcPort,
				Protocol:      v1.ProtocolTCP,
			},
			{
				ContainerPort: restPort,
				Protocol:      v1.ProtocolTCP,
			},
		},
		TerminationMessagePath:   "/dev/termination-log",
		TerminationMessagePolicy: v1.TerminationMessageReadFile,
	}
}

func (pi *PrePackedInitialiser) addTFServerContainer(mlDepSpec *machinelearningv1.SeldonDeploymentSpec, pu *machinelearningv1.PredictiveUnit, deploy *appsv1.Deployment, serverConfig *machinelearningv1.PredictorServerConfig) error {
	ty := machinelearningv1.MODEL
	pu.Type = &ty

	c := utils.GetContainerForDeployment(deploy, pu.Name)

	var tfServingContainer *v1.Container
	if mlDepSpec.Protocol == machinelearningv1.ProtocolTensorflow {
		tfServingContainer = c
	} else {
		c.Image = serverConfig.PrepackImageName(mlDepSpec.Protocol, pu)
		SetUriParamsForTFServingProxyContainer(pu, c)
		tfServingContainer = utils.GetContainerForDeployment(deploy, constants.TFServingContainerName)
	}

	existing := tfServingContainer != nil
	if !existing {
		tfServingContainer = createTensorflowServingContainer(mlDepSpec, pu, mlDepSpec.Protocol == machinelearningv1.ProtocolTensorflow)
		deploy.Spec.Template.Spec.Containers = append(deploy.Spec.Template.Spec.Containers, *tfServingContainer)
	} else {
		// Update any missing fields
		protoType := createTensorflowServingContainer(mlDepSpec, pu, mlDepSpec.Protocol == machinelearningv1.ProtocolTensorflow)
		if tfServingContainer.Image == "" {
			tfServingContainer.Image = protoType.Image
		}
		if tfServingContainer.Args == nil || len(tfServingContainer.Args) == 0 {
			tfServingContainer.Args = protoType.Args
		}
		if tfServingContainer.Ports == nil || len(tfServingContainer.Ports) == 0 {
			tfServingContainer.Ports = protoType.Ports
		}
	}

	envSecretRefName := extractEnvSecretRefName(pu)

	mi := NewModelInitializer(pi.ctx, pi.clientset)
	_, err := mi.InjectModelInitializer(deploy, tfServingContainer.Name, pu.ModelURI, pu.ServiceAccountName, envSecretRefName, pu.StorageInitializerImage)
	if err != nil {
		return err
	}
	return nil
}

func (pi *PrePackedInitialiser) addTritonServer(mlDepSpec *machinelearningv1.SeldonDeploymentSpec, pu *machinelearningv1.PredictiveUnit, deploy *appsv1.Deployment, serverConfig *machinelearningv1.PredictorServerConfig) error {

	c := utils.GetContainerForDeployment(deploy, pu.Name)
	existing := c != nil

	// Define the default arguments
	args := []string{
		"/opt/tritonserver/bin/tritonserver",
		constants.TritonArgGrpcPort + strconv.Itoa(int(pu.Endpoint.GrpcPort)),
		constants.TritonArgHttpPort + strconv.Itoa(int(pu.Endpoint.HttpPort)),
	}

	// Triton can support loading models directory from cloud storage modelURI, enabled with "no-storage-initializer" annotation
	// see: https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md
	noStorage := strings.ToLower(mlDepSpec.Annotations[machinelearningv1.ANNOTATION_NO_STOARGE_INITIALIZER]) == "true"
	if !noStorage {
		args = append(args, constants.TritonArgModelRepository+DefaultModelLocalMountPath)
		args = append(args, constants.TritonArgStrictModelConfig+"false")
	} else {
		args = append(args, constants.TritonArgModelRepository+pu.ModelURI)
		// Optionally allow "model_control_mode=explicit" and one or more "load_model=model_name" parameters
		// see: https://github.com/triton-inference-server/server/blob/main/docs/model_management.md
		for _, paramElement := range pu.Parameters {
			if strings.ToLower(paramElement.Name) == "model_control_mode" {
				args = append(args, constants.TritonArgModelControlMode+paramElement.Value)
			} else if strings.ToLower(paramElement.Name) == "load_model" {
				args = append(args, constants.TritonArgLoadModel+paramElement.Value)
			} else if strings.ToLower(paramElement.Name) == "strict_model_config" {
				args = append(args, constants.TritonArgStrictModelConfig+paramElement.Value)
			}
		}
	}

	cServer := &v1.Container{
		Name: pu.Name,
		Args: args,
		Ports: []v1.ContainerPort{
			{
				Name:          "grpc",
				ContainerPort: pu.Endpoint.GrpcPort,
				Protocol:      v1.ProtocolTCP,
			},
			{
				Name:          "http",
				ContainerPort: pu.Endpoint.HttpPort,
				Protocol:      v1.ProtocolTCP,
			},
		},
		ReadinessProbe: &v1.Probe{
			ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
				Path:   constants.KFServingProbeReadyPath,
				Port:   intstr.FromString("http"),
				Scheme: v1.URISchemeHTTP,
			}},
			InitialDelaySeconds: 20,
			TimeoutSeconds:      1,
			PeriodSeconds:       10,
			SuccessThreshold:    1,
			FailureThreshold:    3,
		},
		LivenessProbe: &v1.Probe{
			ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
				Path:   constants.KFServingProbeLivePath,
				Port:   intstr.FromString("http"),
				Scheme: v1.URISchemeHTTP,
			}},
			InitialDelaySeconds: 60,
			TimeoutSeconds:      1,
			PeriodSeconds:       10,
			SuccessThreshold:    1,
			FailureThreshold:    3,
		},
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      machinelearningv1.PODINFO_VOLUME_NAME,
				MountPath: machinelearningv1.PODINFO_VOLUME_PATH,
			},
		},
		TerminationMessagePath:   "/dev/termination-log",
		TerminationMessagePolicy: v1.TerminationMessageReadFile,
	}
	cServer.Image = serverConfig.PrepackImageName(mlDepSpec.Protocol, pu)

	envSecretRefName := extractEnvSecretRefName(pu)
	if noStorage {
		// Add secrets directly to triton server if not using storage initializer
		addEnvFromSecret(cServer, envSecretRefName)
	}

	if existing {
		// Overwrite core items if not existing or required
		if c.Image == "" {
			c.Image = cServer.Image
		}
		if c.Args == nil {
			c.Args = cServer.Args
		}
		if c.EnvFrom == nil {
			c.EnvFrom = cServer.EnvFrom
		}
		if c.ReadinessProbe == nil {
			c.ReadinessProbe = cServer.ReadinessProbe
		}
		if c.LivenessProbe == nil {
			c.LivenessProbe = cServer.LivenessProbe
		}
		if c.SecurityContext == nil {
			c.SecurityContext = cServer.SecurityContext
		}
		// Ports always overwritten
		// Need to look as we seem to add metrics ports automatically which mean this needs to be done
		c.Ports = cServer.Ports
	} else {
		if len(deploy.Spec.Template.Spec.Containers) > 0 {
			deploy.Spec.Template.Spec.Containers = append(deploy.Spec.Template.Spec.Containers, *cServer)
		} else {
			deploy.Spec.Template.Spec.Containers = []v1.Container{*cServer}
		}
	}

	if !noStorage {
		mi := NewModelInitializer(pi.ctx, pi.clientset)
		_, err := mi.InjectModelInitializer(deploy, c.Name, pu.ModelURI, pu.ServiceAccountName, envSecretRefName, pu.StorageInitializerImage)
		if err != nil {
			return err
		}
	}

	return nil
}

func (pi *PrePackedInitialiser) addMLServerDefault(pu *machinelearningv1.PredictiveUnit, deploy *appsv1.Deployment) error {
	c, err := getMLServerContainer(pu, deploy.Namespace)
	if err != nil {
		return err
	}

	existingContainer := utils.GetContainerForDeployment(deploy, pu.Name)
	if existingContainer != nil {
		c = mergeMLServerContainer(existingContainer, c)
	} else {
		templateSpec := deploy.Spec.Template.Spec
		if len(templateSpec.Containers) == 0 {
			templateSpec.Containers = []v1.Container{}
		}

		templateSpec.Containers = append(templateSpec.Containers, *c)
	}

	envSecretRefName := extractEnvSecretRefName(pu)
	mi := NewModelInitializer(pi.ctx, pi.clientset)

	_, err = mi.InjectModelInitializer(deploy, c.Name, pu.ModelURI, pu.ServiceAccountName, envSecretRefName, pu.StorageInitializerImage)
	if err != nil {
		return err
	}

	return nil
}

func (pi *PrePackedInitialiser) addModelDefaultServers(mlDepSepc *machinelearningv1.SeldonDeploymentSpec, pu *machinelearningv1.PredictiveUnit, deploy *appsv1.Deployment, serverConfig *machinelearningv1.PredictorServerConfig) error {
	ty := machinelearningv1.MODEL
	pu.Type = &ty

	if pu.Endpoint == nil {
		pu.Endpoint = &machinelearningv1.Endpoint{Type: machinelearningv1.REST}
	}
	c := utils.GetContainerForDeployment(deploy, pu.Name)
	existing := c != nil
	if !existing {
		c = &v1.Container{
			Name: pu.Name,
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      machinelearningv1.PODINFO_VOLUME_NAME,
					MountPath: machinelearningv1.PODINFO_VOLUME_PATH,
				},
			},
		}
	}

	if c.Image == "" {
		c.Image = serverConfig.PrepackImageName(mlDepSepc.Protocol, pu)
	}

	// Add parameters envvar - point at mount path because initContainer will download
	params := pu.Parameters
	uriParam := machinelearningv1.Parameter{
		Name:  "model_uri",
		Type:  "STRING",
		Value: DefaultModelLocalMountPath,
	}
	params = append(params, uriParam)
	paramStr, err := json.Marshal(params)
	if err != nil {
		return err
	}

	if len(params) > 0 {
		paramsEnvVar := v1.EnvVar{
			Name:  machinelearningv1.ENV_PREDICTIVE_UNIT_PARAMETERS,
			Value: string(paramStr),
		}
		c.Env = utils.SetEnvVar(c.Env, paramsEnvVar, true)
	}

	// Add container to deployment
	if !existing {
		if len(deploy.Spec.Template.Spec.Containers) > 0 {
			deploy.Spec.Template.Spec.Containers = append(deploy.Spec.Template.Spec.Containers, *c)
		} else {
			deploy.Spec.Template.Spec.Containers = []v1.Container{*c}
		}
	}

	envSecretRefName := extractEnvSecretRefName(pu)

	mi := NewModelInitializer(pi.ctx, pi.clientset)
	_, err = mi.InjectModelInitializer(deploy, c.Name, pu.ModelURI, pu.ServiceAccountName, envSecretRefName, pu.StorageInitializerImage)
	if err != nil {
		return err
	}
	return nil
}

func getPredictiveUnitParameterValue(pu *machinelearningv1.PredictiveUnit, fallback string, names ...string) string {
	for _, param := range pu.Parameters {
		for _, name := range names {
			if strings.EqualFold(param.Name, name) && strings.TrimSpace(param.Value) != "" {
				return strings.TrimSpace(param.Value)
			}
		}
	}
	return fallback
}

func getPredictiveUnitBoolParameter(pu *machinelearningv1.PredictiveUnit, fallback bool, names ...string) bool {
	value := strings.ToLower(getPredictiveUnitParameterValue(pu, "", names...))
	if value == "" {
		return fallback
	}
	return value == "true" || value == "1" || value == "yes"
}

func getPredictiveUnitInt32Parameter(pu *machinelearningv1.PredictiveUnit, fallback int32, names ...string) (int32, error) {
	value := getPredictiveUnitParameterValue(pu, "", names...)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("failed to parse %s as integer: %w", strings.Join(names, "/"), err)
	}
	return int32(parsed), nil
}

type vllmModelSourceKind string

const (
	vllmModelSourceHostPath vllmModelSourceKind = "hostPath"
	vllmModelSourcePVC      vllmModelSourceKind = "pvc"
)

// resolvedVLLMConfig is the single runtime input consumed by the Pod renderer.
type resolvedVLLMConfig struct {
	servedModelName      string
	backendImage         string
	runtimeClassName     string
	modelURI             string
	modelSourceKind      vllmModelSourceKind
	modelHostPath        string
	modelPVCClaimName    string
	gpuResourceName      string
	gpuCount             resource.Quantity
	backendPort          int32
	maxModelLen          string
	maxNumSeqs           string
	gpuMemoryUtilization string
	enforceEager         bool
}

func resolveVLLMConfig(pu *machinelearningv1.PredictiveUnit) (resolvedVLLMConfig, error) {
	legacyModelHostPath := getPredictiveUnitParameterValue(pu, "", "model_host_path")
	config := resolvedVLLMConfig{
		servedModelName:      getPredictiveUnitParameterValue(pu, constants.VLLMDefaultServedModelName, "served_model_name", "vllm_model"),
		backendImage:         getPredictiveUnitParameterValue(pu, constants.VLLMDefaultImage, "vllm_image"),
		runtimeClassName:     getPredictiveUnitParameterValue(pu, constants.VLLMDefaultRuntimeClassName, "runtime_class_name"),
		modelURI:             pu.ModelURI,
		modelHostPath:        legacyModelHostPath,
		gpuResourceName:      getPredictiveUnitParameterValue(pu, constants.VLLMDefaultGPUResourceName, "gpu_resource_name"),
		maxModelLen:          getPredictiveUnitParameterValue(pu, constants.VLLMDefaultMaxModelLen, "max_model_len"),
		maxNumSeqs:           getPredictiveUnitParameterValue(pu, constants.VLLMDefaultMaxNumSeqs, "max_num_seqs"),
		gpuMemoryUtilization: getPredictiveUnitParameterValue(pu, constants.VLLMDefaultGPUMemoryUtilization, "gpu_memory_utilization"),
		enforceEager:         getPredictiveUnitBoolParameter(pu, constants.VLLMDefaultEnforceEager, "enforce_eager"),
	}
	if legacyModelHostPath != "" {
		config.modelSourceKind = vllmModelSourceHostPath
	}

	var err error
	if pu.VLLM != nil && pu.VLLM.Engine != nil && pu.VLLM.Engine.Port > 0 {
		config.backendPort = pu.VLLM.Engine.Port
	} else {
		config.backendPort, err = getPredictiveUnitInt32Parameter(pu, constants.VLLMDefaultHTTPPort, "vllm_port", "vllm_backend_port")
		if err != nil {
			return resolvedVLLMConfig{}, err
		}
	}

	gpuCount := getPredictiveUnitParameterValue(pu, constants.VLLMDefaultGPUCount, "gpu_count")
	if pu.VLLM != nil {
		if pu.VLLM.ServedModelName != "" {
			config.servedModelName = pu.VLLM.ServedModelName
		}
		if pu.VLLM.Image != "" {
			config.backendImage = pu.VLLM.Image
		}
		if pu.VLLM.RuntimeClassName != "" {
			config.runtimeClassName = pu.VLLM.RuntimeClassName
		}
		if pu.VLLM.ModelSource != nil {
			hasHostPath := pu.VLLM.ModelSource.HostPath != nil
			hasPVC := pu.VLLM.ModelSource.PVC != nil
			if hasHostPath && hasPVC {
				return resolvedVLLMConfig{}, fmt.Errorf("vLLM modelSource hostPath and pvc are mutually exclusive")
			}
			if hasHostPath && pu.VLLM.ModelSource.HostPath.Path != "" {
				config.modelSourceKind = vllmModelSourceHostPath
				config.modelHostPath = pu.VLLM.ModelSource.HostPath.Path
				config.modelPVCClaimName = ""
			}
			if hasPVC && pu.VLLM.ModelSource.PVC.ClaimName != "" {
				config.modelSourceKind = vllmModelSourcePVC
				config.modelHostPath = ""
				config.modelPVCClaimName = pu.VLLM.ModelSource.PVC.ClaimName
			}
		}
		if pu.VLLM.GPU != nil {
			if pu.VLLM.GPU.ResourceName != "" {
				config.gpuResourceName = pu.VLLM.GPU.ResourceName
			}
			if pu.VLLM.GPU.Count > 0 {
				gpuCount = strconv.FormatInt(int64(pu.VLLM.GPU.Count), 10)
			}
		}
		if pu.VLLM.Engine != nil {
			if pu.VLLM.Engine.MaxModelLen > 0 {
				config.maxModelLen = strconv.FormatInt(int64(pu.VLLM.Engine.MaxModelLen), 10)
			}
			if pu.VLLM.Engine.MaxNumSeqs > 0 {
				config.maxNumSeqs = strconv.FormatInt(int64(pu.VLLM.Engine.MaxNumSeqs), 10)
			}
			if pu.VLLM.Engine.GPUMemoryUtilizationPercent > 0 {
				config.gpuMemoryUtilization = strconv.FormatFloat(float64(pu.VLLM.Engine.GPUMemoryUtilizationPercent)/100, 'f', -1, 64)
			}
			if pu.VLLM.Engine.EnforceEager != nil {
				config.enforceEager = *pu.VLLM.Engine.EnforceEager
			}
		}
	}

	config.gpuCount, err = resource.ParseQuantity(gpuCount)
	if err != nil {
		return resolvedVLLMConfig{}, fmt.Errorf("failed to parse vLLM gpu_count %q: %w", gpuCount, err)
	}

	return config, nil
}

func addEnvVarIfMissing(c *v1.Container, envVar v1.EnvVar) {
	if !utils.HasEnvVar(c.Env, envVar.Name) {
		c.Env = append(c.Env, envVar)
	}
}

func ensureContainerPort(c *v1.Container, name string, port int32) {
	if machinelearningv1.GetPort(name, c.Ports) == nil {
		c.Ports = append(c.Ports, v1.ContainerPort{
			Name:          name,
			ContainerPort: port,
			Protocol:      v1.ProtocolTCP,
		})
	}
}

func ensureVolumeMount(c *v1.Container, mount v1.VolumeMount) {
	for _, existing := range c.VolumeMounts {
		if existing.Name == mount.Name {
			return
		}
	}
	c.VolumeMounts = append(c.VolumeMounts, mount)
}

func ensureVolume(podSpec *v1.PodSpec, volume v1.Volume) {
	for _, existing := range podSpec.Volumes {
		if existing.Name == volume.Name {
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

func setVLLMProbeDefaults(c *v1.Container) {
	probeHandler := v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
		Path:   "/health",
		Port:   intstr.FromString(constants.VLLMHTTPPortName),
		Scheme: v1.URISchemeHTTP,
	}}

	if c.StartupProbe == nil {
		c.StartupProbe = &v1.Probe{
			ProbeHandler:     probeHandler,
			PeriodSeconds:    10,
			FailureThreshold: 90,
		}
	}
	if c.ReadinessProbe == nil {
		c.ReadinessProbe = &v1.Probe{
			ProbeHandler:     probeHandler,
			PeriodSeconds:    10,
			FailureThreshold: 3,
		}
	}
	if c.LivenessProbe == nil {
		c.LivenessProbe = &v1.Probe{
			ProbeHandler:        probeHandler,
			InitialDelaySeconds: 300,
			PeriodSeconds:       30,
			FailureThreshold:    3,
		}
	}
}

func setVLLMAdapterDefaults(mlDep *machinelearningv1.SeldonDeployment, pu *machinelearningv1.PredictiveUnit, adapter *v1.Container, serverConfig *machinelearningv1.PredictorServerConfig, config resolvedVLLMConfig) {
	if adapter.Image == "" {
		adapter.Image = serverConfig.PrepackImageName(mlDep.Spec.Protocol, pu)
	}
	if adapter.ImagePullPolicy == "" {
		adapter.ImagePullPolicy = v1.PullIfNotPresent
	}

	adapterPort := pu.Endpoint.HttpPort
	if adapterPort == 0 {
		adapterPort = constants.FirstHttpPortNumber
		pu.Endpoint.HttpPort = adapterPort
		pu.Endpoint.ServicePort = adapterPort
	}
	ensureContainerPort(adapter, constants.HttpPortName, adapterPort)

	if adapter.ReadinessProbe == nil {
		adapter.ReadinessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
				Path:   "/ready",
				Port:   intstr.FromString(constants.HttpPortName),
				Scheme: v1.URISchemeHTTP,
			}},
			PeriodSeconds:    10,
			FailureThreshold: 12,
		}
	}
	if adapter.LivenessProbe == nil {
		adapter.LivenessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
				Path:   "/live",
				Port:   intstr.FromString(constants.HttpPortName),
				Scheme: v1.URISchemeHTTP,
			}},
			InitialDelaySeconds: 30,
			PeriodSeconds:       10,
			FailureThreshold:    3,
		}
	}

	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "ADAPTER_HTTP_PORT", Value: strconv.Itoa(int(adapterPort))})
	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "VLLM_BASE_URL", Value: "http://127.0.0.1:" + strconv.Itoa(int(config.backendPort))})
	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "VLLM_MODEL", Value: config.servedModelName})
	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "VLLM_API_KIND", Value: getPredictiveUnitParameterValue(pu, "chat", "vllm_api_kind", "openai_api_kind")})
	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "DEFAULT_MAX_TOKENS", Value: getPredictiveUnitParameterValue(pu, "64", "default_max_tokens")})
	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "REQUEST_TIMEOUT_MS", Value: getPredictiveUnitParameterValue(pu, "60000", "request_timeout_ms")})
	addEnvVarIfMissing(adapter, v1.EnvVar{Name: "DEFAULT_TEMPERATURE", Value: getPredictiveUnitParameterValue(pu, "0.2", "default_temperature")})
	addEnvVarIfMissing(adapter, v1.EnvVar{
		Name: "POD_NAMESPACE",
		ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{
			FieldPath: "metadata.namespace",
		}},
	})

	apiKey := getPredictiveUnitParameterValue(pu, "", "vllm_api_key")
	if apiKey != "" {
		addEnvVarIfMissing(adapter, v1.EnvVar{Name: "VLLM_API_KEY", Value: apiKey})
	}
}

func setVLLMBackendDefaults(pu *machinelearningv1.PredictiveUnit, deploy *appsv1.Deployment, backend *v1.Container, config resolvedVLLMConfig) error {
	if backend.Image == "" {
		backend.Image = config.backendImage
	}
	if backend.ImagePullPolicy == "" {
		backend.ImagePullPolicy = v1.PullIfNotPresent
	}
	ensureContainerPort(backend, constants.VLLMHTTPPortName, config.backendPort)

	if len(backend.Args) == 0 {
		backend.Args = []string{
			"--model",
			config.modelURI,
			"--served-model-name",
			config.servedModelName,
			"--host",
			"0.0.0.0",
			"--port",
			strconv.Itoa(int(config.backendPort)),
			"--max-model-len",
			config.maxModelLen,
			"--gpu-memory-utilization",
			config.gpuMemoryUtilization,
			"--max-num-seqs",
			config.maxNumSeqs,
		}
		if config.enforceEager {
			backend.Args = append(backend.Args, "--enforce-eager")
		}
	}

	if backend.Resources.Limits == nil {
		backend.Resources.Limits = v1.ResourceList{}
	}
	if _, ok := backend.Resources.Limits[v1.ResourceName(config.gpuResourceName)]; !ok {
		backend.Resources.Limits[v1.ResourceName(config.gpuResourceName)] = config.gpuCount
	}

	rootUser := int64(0)
	if backend.SecurityContext == nil {
		backend.SecurityContext = &v1.SecurityContext{}
	}
	if backend.SecurityContext.RunAsUser == nil {
		backend.SecurityContext.RunAsUser = &rootUser
	}

	setVLLMProbeDefaults(backend)

	if deploy.Spec.Template.Spec.RuntimeClassName == nil && config.runtimeClassName != "" {
		deploy.Spec.Template.Spec.RuntimeClassName = &config.runtimeClassName
	}

	if config.modelSourceKind != "" {
		if !strings.HasPrefix(config.modelURI, "/") {
			return fmt.Errorf("vLLM modelUri must be an absolute container path when a model source is set")
		}
		modelVolume := v1.Volume{Name: constants.VLLMModelVolumeName}
		switch config.modelSourceKind {
		case vllmModelSourceHostPath:
			hostPathType := v1.HostPathDirectory
			modelVolume.VolumeSource = v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: config.modelHostPath,
					Type: &hostPathType,
				},
			}
		case vllmModelSourcePVC:
			modelVolume.VolumeSource = v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: config.modelPVCClaimName,
					ReadOnly:  true,
				},
			}
		default:
			return fmt.Errorf("unsupported vLLM model source %q", config.modelSourceKind)
		}
		ensureVolume(&deploy.Spec.Template.Spec, modelVolume)
		ensureVolumeMount(backend, v1.VolumeMount{
			Name:      constants.VLLMModelVolumeName,
			MountPath: config.modelURI,
			ReadOnly:  true,
		})
	}

	shmSize := getPredictiveUnitParameterValue(pu, constants.VLLMDefaultSharedMemorySize, "shm_size")
	shmQuantity, err := resource.ParseQuantity(shmSize)
	if err != nil {
		return fmt.Errorf("failed to parse vLLM shm_size %q: %w", shmSize, err)
	}
	ensureVolume(&deploy.Spec.Template.Spec, v1.Volume{
		Name: constants.VLLMSharedMemoryVolumeName,
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{
				Medium:    v1.StorageMediumMemory,
				SizeLimit: &shmQuantity,
			},
		},
	})
	ensureVolumeMount(backend, v1.VolumeMount{
		Name:      constants.VLLMSharedMemoryVolumeName,
		MountPath: "/dev/shm",
	})

	return nil
}

func (pi *PrePackedInitialiser) addVLLMServer(mlDep *machinelearningv1.SeldonDeployment, pu *machinelearningv1.PredictiveUnit, deploy *appsv1.Deployment, serverConfig *machinelearningv1.PredictorServerConfig) error {
	ty := machinelearningv1.MODEL
	pu.Type = &ty

	if pu.Endpoint == nil {
		pu.Endpoint = &machinelearningv1.Endpoint{Type: machinelearningv1.REST}
	}
	if pu.ModelURI == "" {
		return fmt.Errorf("vLLM modelUri must not be empty")
	}

	config, err := resolveVLLMConfig(pu)
	if err != nil {
		return err
	}
	if deploy.Labels == nil {
		deploy.Labels = map[string]string{}
	}
	if deploy.Spec.Template.Labels == nil {
		deploy.Spec.Template.Labels = map[string]string{}
	}
	deploy.Labels[constants.VLLMRuntimeLabelKey] = constants.VLLMRuntimeLabelValue
	deploy.Spec.Template.Labels[constants.VLLMRuntimeLabelKey] = constants.VLLMRuntimeLabelValue

	backend := utils.GetContainerForDeployment(deploy, constants.VLLMContainerName)
	if backend != nil {
		if port := machinelearningv1.GetPort(constants.VLLMHTTPPortName, backend.Ports); port != nil && port.ContainerPort > 0 {
			config.backendPort = port.ContainerPort
		}
	}

	adapter := utils.GetContainerForDeployment(deploy, pu.Name)
	if adapter == nil {
		adapter = &v1.Container{Name: pu.Name}
		deploy.Spec.Template.Spec.Containers = append(deploy.Spec.Template.Spec.Containers, *adapter)
		adapter = utils.GetContainerForDeployment(deploy, pu.Name)
	}
	setVLLMAdapterDefaults(mlDep, pu, adapter, serverConfig, config)

	if backend == nil {
		backend = &v1.Container{Name: constants.VLLMContainerName}
		deploy.Spec.Template.Spec.Containers = append(deploy.Spec.Template.Spec.Containers, *backend)
		backend = utils.GetContainerForDeployment(deploy, constants.VLLMContainerName)
	}

	return setVLLMBackendDefaults(pu, deploy, backend, config)
}

func SetUriParamsForTFServingProxyContainer(pu *machinelearningv1.PredictiveUnit, c *v1.Container) {

	parameters := pu.Parameters

	hasUriParams := false
	if len(pu.Parameters) > 0 {
		for _, paramElement := range pu.Parameters {
			if paramElement.Name == "rest_endpoint" || paramElement.Name == "grpc_endpoint" {
				hasUriParams = true
			}
		}
	}
	if !hasUriParams {
		uriParam := machinelearningv1.Parameter{
			Name:  "rest_endpoint",
			Type:  "STRING",
			Value: "http://0.0.0.0:2001",
		}
		parameters = append(parameters, uriParam)
		uriParam = machinelearningv1.Parameter{
			Name:  "grpc_endpoint",
			Type:  "STRING",
			Value: "0.0.0.0:2000",
		}
		parameters = append(parameters, uriParam)
	}

	modelNameParam := machinelearningv1.Parameter{
		Name:  "model_name",
		Type:  "STRING",
		Value: pu.Name,
	}

	parameters = append(parameters, modelNameParam)

	if len(parameters) > 0 {
		parametersEnvVar := v1.EnvVar{
			Name:  machinelearningv1.ENV_PREDICTIVE_UNIT_PARAMETERS,
			Value: utils.GetPredictiveUnitAsJson(parameters),
		}
		c.Env = utils.SetEnvVar(c.Env, parametersEnvVar, true)
	}
}

func (pi *PrePackedInitialiser) createStandaloneModelServers(mlDep *machinelearningv1.SeldonDeployment, p *machinelearningv1.PredictorSpec, c *components, pu *machinelearningv1.PredictiveUnit, podSecurityContext *v1.PodSecurityContext) error {

	if machinelearningv1.IsPrepack(pu) {
		sPodSpec, idx := utils.GetSeldonPodSpecForPredictiveUnit(p, pu.Name)
		if sPodSpec == nil {
			return fmt.Errorf("Failed to find PodSpec for Prepackaged server PreditiveUnit named %s", pu.Name)
		}
		depName := machinelearningv1.GetDeploymentName(mlDep, *p, sPodSpec, idx)
		seldonId := machinelearningv1.GetSeldonDeploymentName(mlDep)

		var deploy *appsv1.Deployment
		existing := false
		for i := 0; i < len(c.deployments); i++ {
			d := c.deployments[i]
			if strings.Compare(d.Name, depName) == 0 {
				deploy = d
				existing = true
				break
			}
		}

		// might not be a Deployment yet - if so we have to create one
		if deploy == nil {
			seldonId := machinelearningv1.GetSeldonDeploymentName(mlDep)
			deploy = createDeploymentWithoutEngine(depName, seldonId, sPodSpec, p, mlDep, podSecurityContext, true)
		}

		// apply serviceAccountName to pod to enable EKS fine-grained IAM roles
		if pu.ServiceAccountName != "" {
			deploy.Spec.Template.Spec.ServiceAccountName = pu.ServiceAccountName
		}

		serverConfig := machinelearningv1.GetPrepackServerConfig(string(*pu.Implementation))
		if serverConfig != nil {
			switch *pu.Implementation {
			case machinelearningv1.PrepackTensorflowName:
				if err := pi.addTFServerContainer(&mlDep.Spec, pu, deploy, serverConfig); err != nil {
					return err
				}
			case machinelearningv1.PrepackTritonName:
				if err := pi.addTritonServer(&mlDep.Spec, pu, deploy, serverConfig); err != nil {
					return err
				}
			case machinelearningv1.PrepackVLLMName:
				if err := pi.addVLLMServer(mlDep, pu, deploy, serverConfig); err != nil {
					return err
				}
			default:
				// If protocol is V2, try to add container with MLServer
				if mlDep.Spec.Protocol == machinelearningv1.ProtocolKFServing || mlDep.Spec.Protocol == machinelearningv1.ProtocolV2 {
					err := pi.addMLServerDefault(pu, deploy)
					if err != nil {
						return err
					}
				} else {
					if err := pi.addModelDefaultServers(&mlDep.Spec, pu, deploy, serverConfig); err != nil {
						return err
					}
				}
			}
		} else {
			return fmt.Errorf("Failed to get server config for %s", *pu.Implementation)
		}

		if !existing {

			// this is a new deployment so its containers won't have a containerService
			for k := 0; k < len(deploy.Spec.Template.Spec.Containers); k++ {
				con := &deploy.Spec.Template.Spec.Containers[k]

				//checking for con.Name != "" is a fallback check that we haven't got an empty/nil container as name is required
				if con.Name != EngineContainerName && con.Name != constants.TFServingContainerName && con.Name != "" {
					svc := createContainerService(deploy, *p, mlDep, con, *c, seldonId)
					if svc != nil {
						c.services = append(c.services, svc)
					}
				}
			}
			if len(deploy.Spec.Template.Spec.Containers) > 0 && deploy.Spec.Template.Spec.Containers[0].Name != "" {
				// Add deployment, provided we have a non-empty spec
				c.deployments = append(c.deployments, deploy)
			}
		}
	}

	for i := 0; i < len(pu.Children); i++ {
		if err := pi.createStandaloneModelServers(mlDep, p, c, &pu.Children[i], podSecurityContext); err != nil {
			return err
		}
	}
	return nil
}
