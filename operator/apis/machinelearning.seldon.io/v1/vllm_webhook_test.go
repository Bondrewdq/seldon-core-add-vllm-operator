package v1

import (
	"testing"

	. "github.com/onsi/gomega"
	"github.com/seldonio/seldon-core/operator/constants"
	corev1 "k8s.io/api/core/v1"
)

func newValidVLLMPredictiveUnit(name string) PredictiveUnit {
	implementation := PredictiveUnitImplementation(PrepackVLLMName)
	return PredictiveUnit{
		Name:           name,
		Implementation: &implementation,
		ModelURI:       "/models/qwen",
		VLLM: &VLLMSpec{
			ServedModelName: "qwen-0.5b",
			ModelSource: &VLLMModelSource{
				HostPath: &VLLMHostPathSource{
					Path: "/var/lib/models/qwen",
				},
			},
		},
	}
}

func newValidVLLMSeldonDeploymentSpec() *SeldonDeploymentSpec {
	unit := newValidVLLMPredictiveUnit("llm")
	return &SeldonDeploymentSpec{
		Predictors: []PredictorSpec{
			{
				Name:  "default",
				Graph: unit,
				ComponentSpecs: []*SeldonPodSpec{
					{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  unit.Name,
									Image: "localhost/vllm-adapter:test",
								},
							},
						},
					},
				},
			},
		},
	}
}

func TestDefaultVLLMSpec(t *testing.T) {
	g := NewGomegaWithT(t)
	spec := &VLLMSpec{}

	defaultVLLMSpec(spec)

	g.Expect(spec.ServedModelName).To(BeEmpty())
	g.Expect(spec.ModelSource).To(BeNil())
	g.Expect(spec.Image).To(Equal(constants.VLLMDefaultImage))
	g.Expect(spec.RuntimeClassName).To(Equal(constants.VLLMDefaultRuntimeClassName))
	g.Expect(spec.GPU).ToNot(BeNil())
	g.Expect(spec.GPU.ResourceName).To(Equal(constants.VLLMDefaultGPUResourceName))
	g.Expect(spec.GPU.Count).To(Equal(constants.VLLMDefaultGPUCountValue))
	g.Expect(spec.Engine).ToNot(BeNil())
	g.Expect(spec.Engine.Port).To(Equal(constants.VLLMDefaultHTTPPort))
	g.Expect(spec.Engine.MaxModelLen).To(Equal(constants.VLLMDefaultMaxModelLenValue))
	g.Expect(spec.Engine.MaxNumSeqs).To(Equal(constants.VLLMDefaultMaxNumSeqsValue))
	g.Expect(spec.Engine.GPUMemoryUtilizationPercent).To(Equal(constants.VLLMDefaultGPUMemoryPercent))
	g.Expect(spec.Engine.EnforceEager).ToNot(BeNil())
	g.Expect(*spec.Engine.EnforceEager).To(Equal(constants.VLLMDefaultEnforceEager))
}

func TestDefaultVLLMSpecPreservesExplicitValues(t *testing.T) {
	g := NewGomegaWithT(t)
	enforceEager := false
	unit := PredictiveUnit{
		VLLM: &VLLMSpec{
			ServedModelName:  "custom-model",
			Image:            "registry.example.com/vllm:v1",
			RuntimeClassName: "custom-runtime",
			ModelSource: &VLLMModelSource{
				HostPath: &VLLMHostPathSource{Path: "/srv/models/custom"},
			},
			GPU: &VLLMGPUSpec{
				ResourceName: "example.com/gpu",
				Count:        2,
			},
			Engine: &VLLMEngineSpec{
				Port:                        9000,
				MaxModelLen:                 2048,
				MaxNumSeqs:                  4,
				GPUMemoryUtilizationPercent: 80,
				EnforceEager:                &enforceEager,
			},
		},
	}
	want := unit.DeepCopy().VLLM

	defaultVLLMSpec(unit.VLLM)

	g.Expect(unit.VLLM).To(Equal(want))
	g.Expect(*unit.VLLM.Engine.EnforceEager).To(BeFalse())
}

func TestVLLMDefaultsAreRecursiveIdempotentAndLegacyCompatible(t *testing.T) {
	g := NewGomegaWithT(t)
	implementation := PredictiveUnitImplementation(PrepackVLLMName)
	root := newValidVLLMPredictiveUnit("root")
	root.Children = []PredictiveUnit{
		newValidVLLMPredictiveUnit("child"),
		{
			Name:           "legacy",
			Implementation: &implementation,
			ModelURI:       "/models/legacy",
			Parameters: []Parameter{
				{Name: "served_model_name", Value: "legacy-model", Type: STRING},
			},
		},
	}

	addDefaultsToGraph(&root)
	g.Expect(root.VLLM.Engine).ToNot(BeNil())
	g.Expect(root.Children[0].VLLM.Engine).ToNot(BeNil())
	g.Expect(root.Children[1].VLLM).To(BeNil())
	want := root.DeepCopy()

	addDefaultsToGraph(&root)

	g.Expect(&root).To(Equal(want))
}

func TestVLLMDefaultsDoNotInferImplementation(t *testing.T) {
	g := NewGomegaWithT(t)
	unit := PredictiveUnit{VLLM: &VLLMSpec{}}

	addDefaultsToGraph(&unit)

	g.Expect(unit.Implementation).ToNot(BeNil())
	g.Expect(*unit.Implementation).To(Equal(UNKNOWN_IMPLEMENTATION))
	g.Expect(*unit.Implementation).ToNot(Equal(PredictiveUnitImplementation(PrepackVLLMName)))
}

func TestValidateTypedVLLMConfiguration(t *testing.T) {
	g := NewGomegaWithT(t)
	spec := newValidVLLMSeldonDeploymentSpec()

	err := spec.ValidateSeldonDeployment()

	g.Expect(err).ToNot(HaveOccurred())
}

func TestValidateTypedVLLMConfigurationErrors(t *testing.T) {
	tests := []struct {
		name         string
		mutate       func(*PredictiveUnit)
		expectedPath string
	}{
		{
			name: "implementation mismatch",
			mutate: func(unit *PredictiveUnit) {
				implementation := PredictiveUnitImplementation(PrepackSklearnName)
				unit.Implementation = &implementation
			},
			expectedPath: "spec.predictors[0].graph.implementation",
		},
		{
			name: "blank served model name",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.ServedModelName = "   "
			},
			expectedPath: "spec.predictors[0].graph.vllm.servedModelName",
		},
		{
			name: "missing model uri",
			mutate: func(unit *PredictiveUnit) {
				unit.ModelURI = ""
			},
			expectedPath: "spec.predictors[0].graph.modelUri",
		},
		{
			name: "relative model uri",
			mutate: func(unit *PredictiveUnit) {
				unit.ModelURI = "models/qwen"
			},
			expectedPath: "spec.predictors[0].graph.modelUri",
		},
		{
			name: "missing model source",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.ModelSource = nil
			},
			expectedPath: "spec.predictors[0].graph.vllm.modelSource",
		},
		{
			name: "missing host path source",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.ModelSource.HostPath = nil
			},
			expectedPath: "spec.predictors[0].graph.vllm.modelSource.hostPath",
		},
		{
			name: "relative host path",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.ModelSource.HostPath.Path = "models/qwen"
			},
			expectedPath: "spec.predictors[0].graph.vllm.modelSource.hostPath.path",
		},
		{
			name: "image containing whitespace",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.Image = "vllm/image:bad tag"
			},
			expectedPath: "spec.predictors[0].graph.vllm.image",
		},
		{
			name: "invalid runtime class",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.RuntimeClassName = "NVIDIA_GPU"
			},
			expectedPath: "spec.predictors[0].graph.vllm.runtimeClassName",
		},
		{
			name: "unqualified GPU resource name",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.GPU = &VLLMGPUSpec{ResourceName: "gpu", Count: 1}
			},
			expectedPath: "spec.predictors[0].graph.vllm.gpu.resourceName",
		},
		{
			name: "reserved GPU resource name",
			mutate: func(unit *PredictiveUnit) {
				unit.VLLM.GPU = &VLLMGPUSpec{ResourceName: "kubernetes.io/gpu", Count: 1}
			},
			expectedPath: "spec.predictors[0].graph.vllm.gpu.resourceName",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			g := NewGomegaWithT(t)
			spec := newValidVLLMSeldonDeploymentSpec()
			test.mutate(&spec.Predictors[0].Graph)

			err := spec.ValidateSeldonDeployment()

			g.Expect(err).To(HaveOccurred())
			g.Expect(err.Error()).To(ContainSubstring(test.expectedPath))
		})
	}
}

func TestValidateTypedVLLMChildReportsExactPath(t *testing.T) {
	g := NewGomegaWithT(t)
	spec := newValidVLLMSeldonDeploymentSpec()
	child := newValidVLLMPredictiveUnit("child")
	child.VLLM.ModelSource.HostPath.Path = "relative/path"
	spec.Predictors[0].Graph.Children = []PredictiveUnit{child}
	spec.Predictors[0].ComponentSpecs[0].Spec.Containers = append(
		spec.Predictors[0].ComponentSpecs[0].Spec.Containers,
		corev1.Container{Name: child.Name, Image: "localhost/vllm-adapter:test"},
	)

	err := spec.ValidateSeldonDeployment()

	g.Expect(err).To(HaveOccurred())
	g.Expect(err.Error()).To(ContainSubstring(
		"spec.predictors[0].graph.children[0].vllm.modelSource.hostPath.path",
	))
}

func TestValidateLegacyVLLMParametersRemainSupported(t *testing.T) {
	g := NewGomegaWithT(t)
	spec := newValidVLLMSeldonDeploymentSpec()
	unit := &spec.Predictors[0].Graph
	unit.VLLM = nil
	unit.Parameters = []Parameter{
		{Name: "served_model_name", Value: "legacy-model", Type: STRING},
		{Name: "gpu_count", Value: "1", Type: INT},
	}

	err := spec.ValidateSeldonDeployment()

	g.Expect(err).ToNot(HaveOccurred())
}

func TestValidateTypedVLLMAllowsLegacyParametersDuringMigration(t *testing.T) {
	g := NewGomegaWithT(t)
	spec := newValidVLLMSeldonDeploymentSpec()
	spec.Predictors[0].Graph.Parameters = []Parameter{
		{Name: "served_model_name", Value: "legacy-model", Type: STRING},
	}

	err := spec.ValidateSeldonDeployment()

	g.Expect(err).ToNot(HaveOccurred())
}
