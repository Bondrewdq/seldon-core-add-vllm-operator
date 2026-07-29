package controllers

import (
	"testing"

	machinelearningv1 "github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
	"github.com/seldonio/seldon-core/operator/constants"
	"github.com/seldonio/seldon-core/operator/utils"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestResolveVLLMConfigUsesLegacyParameters(t *testing.T) {
	pu := &machinelearningv1.PredictiveUnit{
		ModelURI: "/models/legacy",
		Parameters: []machinelearningv1.Parameter{
			{Name: "served_model_name", Value: "legacy-model"},
			{Name: "vllm_image", Value: "example.com/vllm:legacy"},
			{Name: "runtime_class_name", Value: "legacy-runtime"},
			{Name: "model_host_path", Value: "/node/legacy"},
			{Name: "gpu_resource_name", Value: "example.com/legacy-gpu"},
			{Name: "gpu_count", Value: "2"},
			{Name: "vllm_port", Value: "8181"},
			{Name: "max_model_len", Value: "2048"},
			{Name: "max_num_seqs", Value: "4"},
			{Name: "gpu_memory_utilization", Value: "0.75"},
			{Name: "enforce_eager", Value: "false"},
		},
	}

	config, err := resolveVLLMConfig(pu)
	if err != nil {
		t.Fatalf("resolveVLLMConfig() error = %v", err)
	}

	assertResolvedVLLMConfig(t, config, resolvedVLLMConfig{
		servedModelName:      "legacy-model",
		backendImage:         "example.com/vllm:legacy",
		runtimeClassName:     "legacy-runtime",
		modelURI:             "/models/legacy",
		modelHostPath:        "/node/legacy",
		gpuResourceName:      "example.com/legacy-gpu",
		gpuCount:             resource.MustParse("2"),
		backendPort:          8181,
		maxModelLen:          "2048",
		maxNumSeqs:           "4",
		gpuMemoryUtilization: "0.75",
		enforceEager:         false,
	})
}

func TestResolveVLLMConfigPrefersTypedFields(t *testing.T) {
	enforceEager := false
	pu := &machinelearningv1.PredictiveUnit{
		ModelURI: "/models/typed",
		Parameters: []machinelearningv1.Parameter{
			{Name: "served_model_name", Value: "legacy-model"},
			{Name: "vllm_image", Value: "example.com/vllm:legacy"},
			{Name: "runtime_class_name", Value: "legacy-runtime"},
			{Name: "model_host_path", Value: "/node/legacy"},
			{Name: "gpu_resource_name", Value: "example.com/legacy-gpu"},
			{Name: "gpu_count", Value: "invalid-legacy-value"},
			{Name: "vllm_port", Value: "invalid-legacy-value"},
			{Name: "max_model_len", Value: "2048"},
			{Name: "max_num_seqs", Value: "4"},
			{Name: "gpu_memory_utilization", Value: "0.75"},
			{Name: "enforce_eager", Value: "true"},
		},
		VLLM: &machinelearningv1.VLLMSpec{
			ServedModelName:  "typed-model",
			Image:            "example.com/vllm:typed",
			RuntimeClassName: "typed-runtime",
			ModelSource: &machinelearningv1.VLLMModelSource{
				HostPath: &machinelearningv1.VLLMHostPathSource{Path: "/node/typed"},
			},
			GPU: &machinelearningv1.VLLMGPUSpec{
				ResourceName: "example.com/typed-gpu",
				Count:        3,
			},
			Engine: &machinelearningv1.VLLMEngineSpec{
				Port:                        8282,
				MaxModelLen:                 4096,
				MaxNumSeqs:                  8,
				GPUMemoryUtilizationPercent: 65,
				EnforceEager:                &enforceEager,
			},
		},
	}

	config, err := resolveVLLMConfig(pu)
	if err != nil {
		t.Fatalf("resolveVLLMConfig() error = %v", err)
	}

	assertResolvedVLLMConfig(t, config, resolvedVLLMConfig{
		servedModelName:      "typed-model",
		backendImage:         "example.com/vllm:typed",
		runtimeClassName:     "typed-runtime",
		modelURI:             "/models/typed",
		modelHostPath:        "/node/typed",
		gpuResourceName:      "example.com/typed-gpu",
		gpuCount:             resource.MustParse("3"),
		backendPort:          8282,
		maxModelLen:          "4096",
		maxNumSeqs:           "8",
		gpuMemoryUtilization: "0.65",
		enforceEager:         false,
	})
}

func TestAddVLLMServerRendersTypedConfiguration(t *testing.T) {
	enforceEager := false
	pu := newTypedVLLMPredictiveUnit(&enforceEager)
	mlDep := &machinelearningv1.SeldonDeployment{}
	deploy := &appsv1.Deployment{}
	serverConfig := testVLLMServerConfig()

	initializer := &PrePackedInitialiser{}
	if err := initializer.addVLLMServer(mlDep, pu, deploy, serverConfig); err != nil {
		t.Fatalf("addVLLMServer() error = %v", err)
	}

	if deploy.Spec.Template.Spec.RuntimeClassName == nil || *deploy.Spec.Template.Spec.RuntimeClassName != "typed-runtime" {
		t.Fatalf("runtimeClassName = %v, want typed-runtime", deploy.Spec.Template.Spec.RuntimeClassName)
	}

	adapter := utils.GetContainerForDeployment(deploy, pu.Name)
	if adapter == nil {
		t.Fatal("adapter container was not generated")
	}
	assertEnvValue(t, adapter, "VLLM_BASE_URL", "http://127.0.0.1:8282")
	assertEnvValue(t, adapter, "VLLM_MODEL", "typed-model")

	backend := utils.GetContainerForDeployment(deploy, constants.VLLMContainerName)
	if backend == nil {
		t.Fatal("vLLM backend container was not generated")
	}
	if backend.Image != "example.com/vllm:typed" {
		t.Fatalf("backend image = %q, want example.com/vllm:typed", backend.Image)
	}
	assertContainerPort(t, backend, constants.VLLMHTTPPortName, 8282)
	assertArgValue(t, backend.Args, "--model", "/models/typed")
	assertArgValue(t, backend.Args, "--served-model-name", "typed-model")
	assertArgValue(t, backend.Args, "--port", "8282")
	assertArgValue(t, backend.Args, "--max-model-len", "4096")
	assertArgValue(t, backend.Args, "--max-num-seqs", "8")
	assertArgValue(t, backend.Args, "--gpu-memory-utilization", "0.65")
	if containsString(backend.Args, "--enforce-eager") {
		t.Fatal("backend args contain --enforce-eager for enforceEager=false")
	}

	gpuQuantity, ok := backend.Resources.Limits[corev1.ResourceName("example.com/typed-gpu")]
	if !ok || gpuQuantity.Cmp(resource.MustParse("3")) != 0 {
		t.Fatalf("GPU limit = %v, want example.com/typed-gpu=3", backend.Resources.Limits)
	}
	assertHostPathVolume(t, deploy, constants.VLLMModelVolumeName, "/node/typed")
	assertVolumeMount(t, backend, constants.VLLMModelVolumeName, "/models/typed")
}

func TestAddVLLMServerPreservesExplicitComponentSpecFields(t *testing.T) {
	enforceEager := true
	pu := newTypedVLLMPredictiveUnit(&enforceEager)
	runtimeClassName := "component-runtime"
	explicitGPUQuantity := resource.MustParse("5")
	hostPathType := corev1.HostPathDirectory
	deploy := &appsv1.Deployment{
		Spec: appsv1.DeploymentSpec{Template: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
			RuntimeClassName: &runtimeClassName,
			Containers: []corev1.Container{
				{
					Name:  pu.Name,
					Image: "example.com/adapter:component",
					Env:   []corev1.EnvVar{{Name: "VLLM_MODEL", Value: "component-model"}},
				},
				{
					Name:  constants.VLLMContainerName,
					Image: "example.com/vllm:component",
					Args:  []string{"--component-managed-args"},
					Ports: []corev1.ContainerPort{{Name: constants.VLLMHTTPPortName, ContainerPort: 8383}},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
						corev1.ResourceName("example.com/typed-gpu"): explicitGPUQuantity,
					}},
					VolumeMounts: []corev1.VolumeMount{{
						Name:      constants.VLLMModelVolumeName,
						MountPath: "/component/models",
					}},
				},
			},
			Volumes: []corev1.Volume{{
				Name: constants.VLLMModelVolumeName,
				VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{
					Path: "/node/component",
					Type: &hostPathType,
				}},
			}},
		}}},
	}

	initializer := &PrePackedInitialiser{}
	if err := initializer.addVLLMServer(&machinelearningv1.SeldonDeployment{}, pu, deploy, testVLLMServerConfig()); err != nil {
		t.Fatalf("addVLLMServer() error = %v", err)
	}

	if deploy.Spec.Template.Spec.RuntimeClassName == nil || *deploy.Spec.Template.Spec.RuntimeClassName != "component-runtime" {
		t.Fatalf("runtimeClassName = %v, want component-runtime", deploy.Spec.Template.Spec.RuntimeClassName)
	}
	adapter := utils.GetContainerForDeployment(deploy, pu.Name)
	if adapter.Image != "example.com/adapter:component" {
		t.Fatalf("adapter image = %q, want explicit component image", adapter.Image)
	}
	assertEnvValue(t, adapter, "VLLM_MODEL", "component-model")
	assertEnvValue(t, adapter, "VLLM_BASE_URL", "http://127.0.0.1:8383")

	backend := utils.GetContainerForDeployment(deploy, constants.VLLMContainerName)
	if backend.Image != "example.com/vllm:component" {
		t.Fatalf("backend image = %q, want explicit component image", backend.Image)
	}
	if len(backend.Args) != 1 || backend.Args[0] != "--component-managed-args" {
		t.Fatalf("backend args = %v, want explicit component args", backend.Args)
	}
	assertContainerPort(t, backend, constants.VLLMHTTPPortName, 8383)
	actualGPUQuantity := backend.Resources.Limits[corev1.ResourceName("example.com/typed-gpu")]
	if actualGPUQuantity.Cmp(explicitGPUQuantity) != 0 {
		t.Fatalf("GPU limit = %v, want explicit component limit", backend.Resources.Limits)
	}
	assertHostPathVolume(t, deploy, constants.VLLMModelVolumeName, "/node/component")
	assertVolumeMount(t, backend, constants.VLLMModelVolumeName, "/component/models")
}

func newTypedVLLMPredictiveUnit(enforceEager *bool) *machinelearningv1.PredictiveUnit {
	return &machinelearningv1.PredictiveUnit{
		Name:     "llm-adapter",
		ModelURI: "/models/typed",
		Endpoint: &machinelearningv1.Endpoint{Type: machinelearningv1.REST, HttpPort: 9000},
		VLLM: &machinelearningv1.VLLMSpec{
			ServedModelName:  "typed-model",
			Image:            "example.com/vllm:typed",
			RuntimeClassName: "typed-runtime",
			ModelSource: &machinelearningv1.VLLMModelSource{
				HostPath: &machinelearningv1.VLLMHostPathSource{Path: "/node/typed"},
			},
			GPU: &machinelearningv1.VLLMGPUSpec{
				ResourceName: "example.com/typed-gpu",
				Count:        3,
			},
			Engine: &machinelearningv1.VLLMEngineSpec{
				Port:                        8282,
				MaxModelLen:                 4096,
				MaxNumSeqs:                  8,
				GPUMemoryUtilizationPercent: 65,
				EnforceEager:                enforceEager,
			},
		},
	}
}

func testVLLMServerConfig() *machinelearningv1.PredictorServerConfig {
	return &machinelearningv1.PredictorServerConfig{Protocols: map[machinelearningv1.Protocol]machinelearningv1.PredictorImageConfig{
		machinelearningv1.ProtocolSeldon: {
			ContainerImage:      "example.com/adapter",
			DefaultImageVersion: "test",
		},
	}}
}

func assertResolvedVLLMConfig(t *testing.T, got, want resolvedVLLMConfig) {
	t.Helper()
	if got.servedModelName != want.servedModelName ||
		got.backendImage != want.backendImage ||
		got.runtimeClassName != want.runtimeClassName ||
		got.modelURI != want.modelURI ||
		got.modelHostPath != want.modelHostPath ||
		got.gpuResourceName != want.gpuResourceName ||
		got.backendPort != want.backendPort ||
		got.maxModelLen != want.maxModelLen ||
		got.maxNumSeqs != want.maxNumSeqs ||
		got.gpuMemoryUtilization != want.gpuMemoryUtilization ||
		got.enforceEager != want.enforceEager ||
		got.gpuCount.Cmp(want.gpuCount) != 0 {
		t.Fatalf("resolved config = %#v, want %#v", got, want)
	}
}

func assertEnvValue(t *testing.T, container *corev1.Container, name, want string) {
	t.Helper()
	for _, env := range container.Env {
		if env.Name == name {
			if env.Value != want {
				t.Fatalf("env %s = %q, want %q", name, env.Value, want)
			}
			return
		}
	}
	t.Fatalf("env %s was not generated", name)
}

func assertContainerPort(t *testing.T, container *corev1.Container, name string, want int32) {
	t.Helper()
	port := machinelearningv1.GetPort(name, container.Ports)
	if port == nil || port.ContainerPort != want {
		t.Fatalf("container port %s = %v, want %d", name, port, want)
	}
}

func assertArgValue(t *testing.T, args []string, name, want string) {
	t.Helper()
	for i := 0; i < len(args)-1; i++ {
		if args[i] == name {
			if args[i+1] != want {
				t.Fatalf("argument %s = %q, want %q", name, args[i+1], want)
			}
			return
		}
	}
	t.Fatalf("argument %s was not generated in %v", name, args)
}

func assertHostPathVolume(t *testing.T, deploy *appsv1.Deployment, name, wantPath string) {
	t.Helper()
	for _, volume := range deploy.Spec.Template.Spec.Volumes {
		if volume.Name == name {
			if volume.HostPath == nil || volume.HostPath.Path != wantPath {
				t.Fatalf("hostPath volume %s = %#v, want path %q", name, volume.HostPath, wantPath)
			}
			return
		}
	}
	t.Fatalf("volume %s was not generated", name)
}

func assertVolumeMount(t *testing.T, container *corev1.Container, name, wantPath string) {
	t.Helper()
	for _, mount := range container.VolumeMounts {
		if mount.Name == name {
			if mount.MountPath != wantPath {
				t.Fatalf("volume mount %s = %q, want %q", name, mount.MountPath, wantPath)
			}
			return
		}
	}
	t.Fatalf("volume mount %s was not generated", name)
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
