/*
Copyright 2019 The Seldon Authors.

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

package v1

import (
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/seldonio/seldon-core/operator/constants"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
)

var (
	// log is for logging in this package.
	seldondeploymentLog                 = logf.Log.WithName("seldondeployment")
	ControllerNamespace                 = GetEnv("POD_NAMESPACE", "seldon-system")
	C                                   client.Client
	envPredictiveUnitHttpServicePort    = os.Getenv(ENV_PREDICTIVE_UNIT_HTTP_SERVICE_PORT)
	envPredictiveUnitGrpcServicePort    = os.Getenv(ENV_PREDICTIVE_UNIT_GRPC_SERVICE_PORT)
	envPredictiveUnitServicePortMetrics = os.Getenv(ENV_PREDICTIVE_UNIT_SERVICE_PORT_METRICS)
	envPredictiveUnitMetricsPortName    = GetEnv(ENV_PREDICTIVE_UNIT_METRICS_PORT_NAME, constants.DefaultMetricsPortName)
)

// Get an environment variable given by key or return the fallback.
func GetEnv(key, fallback string) string {
	if value, ok := os.LookupEnv(key); ok {
		return value
	}
	return fallback
}

func (r *SeldonDeployment) SetupWebhookWithManager(mgr ctrl.Manager) error {
	C = mgr.GetClient()
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		Complete()
}

func GetContainerForPredictiveUnit(p *PredictorSpec, name string) *corev1.Container {
	for j := 0; j < len(p.ComponentSpecs); j++ {
		cSpec := p.ComponentSpecs[j]
		for k := 0; k < len(cSpec.Spec.Containers); k++ {
			c := &cSpec.Spec.Containers[k]
			if c.Name == name {
				return c
			}
		}
	}
	return nil
}

func GetComponentSpecIdxForPredictiveUnit(p *PredictorSpec, name string) int {
	for j := 0; j < len(p.ComponentSpecs); j++ {
		cSpec := p.ComponentSpecs[j]
		for k := 0; k < len(cSpec.Spec.Containers); k++ {
			c := &cSpec.Spec.Containers[k]
			if c.Name == name {
				return j
			}
		}
	}
	return 0
}

// --- Validating

func validateVLLMSpec(pu *PredictiveUnit, fldPath *field.Path, allErrs field.ErrorList) field.ErrorList {
	if pu.VLLM == nil {
		return allErrs
	}

	vllmPath := fldPath.Child("vllm")
	implementation := ""
	if pu.Implementation != nil {
		implementation = string(*pu.Implementation)
	}
	if implementation != PrepackVLLMName {
		allErrs = append(allErrs, field.NotSupported(
			fldPath.Child("implementation"),
			implementation,
			[]string{PrepackVLLMName},
		))
	}

	if strings.TrimSpace(pu.VLLM.ServedModelName) == "" {
		allErrs = append(allErrs, field.Required(
			vllmPath.Child("servedModelName"),
			"servedModelName is required for typed VLLM configuration",
		))
	}

	if strings.TrimSpace(pu.ModelURI) == "" {
		allErrs = append(allErrs, field.Required(
			fldPath.Child("modelUri"),
			"modelUri is required for typed VLLM configuration",
		))
	} else if strings.TrimSpace(pu.ModelURI) != pu.ModelURI || !path.IsAbs(pu.ModelURI) {
		allErrs = append(allErrs, field.Invalid(
			fldPath.Child("modelUri"),
			pu.ModelURI,
			"must be an absolute container path without surrounding whitespace",
		))
	}

	modelSourcePath := vllmPath.Child("modelSource")
	if pu.VLLM.ModelSource == nil {
		allErrs = append(allErrs, field.Required(
			modelSourcePath,
			"modelSource is required for typed VLLM configuration",
		))
	} else {
		hasHostPath := pu.VLLM.ModelSource.HostPath != nil
		hasPVC := pu.VLLM.ModelSource.PVC != nil
		switch {
		case !hasHostPath && !hasPVC:
			allErrs = append(allErrs, field.Required(
				modelSourcePath,
				"exactly one of hostPath or pvc is required",
			))
		case hasHostPath && hasPVC:
			allErrs = append(allErrs, field.Invalid(
				modelSourcePath,
				pu.VLLM.ModelSource,
				"hostPath and pvc are mutually exclusive",
			))
		case hasHostPath:
			hostPath := pu.VLLM.ModelSource.HostPath.Path
			if strings.TrimSpace(hostPath) == "" {
				allErrs = append(allErrs, field.Required(
					modelSourcePath.Child("hostPath").Child("path"),
					"hostPath.path is required",
				))
			} else if strings.TrimSpace(hostPath) != hostPath || !path.IsAbs(hostPath) {
				allErrs = append(allErrs, field.Invalid(
					modelSourcePath.Child("hostPath").Child("path"),
					hostPath,
					"must be an absolute node path without surrounding whitespace",
				))
			}
		case hasPVC:
			claimName := pu.VLLM.ModelSource.PVC.ClaimName
			claimNamePath := modelSourcePath.Child("pvc").Child("claimName")
			if strings.TrimSpace(claimName) == "" {
				allErrs = append(allErrs, field.Required(
					claimNamePath,
					"pvc.claimName is required",
				))
			} else if strings.TrimSpace(claimName) != claimName {
				allErrs = append(allErrs, field.Invalid(
					claimNamePath,
					claimName,
					"must not contain surrounding whitespace",
				))
			} else if messages := k8svalidation.IsDNS1123Subdomain(claimName); len(messages) > 0 {
				allErrs = append(allErrs, field.Invalid(
					claimNamePath,
					claimName,
					strings.Join(messages, "; "),
				))
			}
		}
	}

	if pu.VLLM.Image != "" && strings.ContainsAny(pu.VLLM.Image, " \t\r\n") {
		allErrs = append(allErrs, field.Invalid(
			vllmPath.Child("image"),
			pu.VLLM.Image,
			"must not contain whitespace",
		))
	}

	if pu.VLLM.RuntimeClassName != "" {
		if messages := k8svalidation.IsDNS1123Subdomain(pu.VLLM.RuntimeClassName); len(messages) > 0 {
			allErrs = append(allErrs, field.Invalid(
				vllmPath.Child("runtimeClassName"),
				pu.VLLM.RuntimeClassName,
				strings.Join(messages, "; "),
			))
		}
	}

	if pu.VLLM.GPU != nil && pu.VLLM.GPU.ResourceName != "" {
		resourceName := pu.VLLM.GPU.ResourceName
		qualifiedNameErrors := k8svalidation.IsQualifiedName(resourceName)
		parts := strings.SplitN(resourceName, "/", 2)
		reservedPrefix := len(parts) == 2 && (parts[0] == "kubernetes.io" || strings.HasSuffix(parts[0], ".kubernetes.io"))
		if len(parts) != 2 || len(qualifiedNameErrors) > 0 || reservedPrefix {
			allErrs = append(allErrs, field.Invalid(
				vllmPath.Child("gpu").Child("resourceName"),
				resourceName,
				"must be a qualified, non-kubernetes.io extended resource name such as nvidia.com/gpu",
			))
		}
	}

	return allErrs
}

// Check the predictive units to ensure the graph matches up with defined containers.
func (r *SeldonDeploymentSpec) checkPredictiveUnits(pu *PredictiveUnit, p *PredictorSpec, fldPath *field.Path, allErrs field.ErrorList) field.ErrorList {
	allErrs = validateVLLMSpec(pu, fldPath, allErrs)

	if pu.Implementation == nil || *pu.Implementation == UNKNOWN_IMPLEMENTATION {

		if GetContainerForPredictiveUnit(p, pu.Name) == nil {
			allErrs = append(allErrs, field.Invalid(fldPath, pu.Name, "Can't find container for Predictive Unit"))
		}

		if pu.Type != nil && *pu.Type == UNKNOWN_TYPE && (pu.Methods == nil || len(*pu.Methods) == 0) {
			allErrs = append(allErrs, field.Invalid(fldPath, pu.Name, "Predictive Unit has no implementation methods defined. Change to a known type or add what methods it defines"))
		}

	} else if IsPrepack(pu) {
		// Only HuggingFace server is allowed for no ModelURI as it can load from Hub
		if pu.ModelURI == "" && (pu.Implementation == nil || *pu.Implementation != PrepackHuggingFaceName) {
			allErrs = append(allErrs, field.Invalid(fldPath, pu.Name, "Predictive unit modelUri required when using standalone servers"))
		}
		c := GetContainerForPredictiveUnit(p, pu.Name)

		//Current non tensorflow serving prepack servers can not handle tensorflow protocol
		if r.Protocol == ProtocolTensorflow && (*pu.Implementation == PrepackSklearnName || *pu.Implementation == PrepackXGBoostName || *pu.Implementation == PrepackMLFlowName || *pu.Implementation == PrepackHuggingFaceName || *pu.Implementation == PrepackVLLMName) {
			allErrs = append(allErrs, field.Invalid(fldPath, pu.Name, "Prepackaged server does not handle tensorflow protocol "+string(*pu.Implementation)))
		}

		if c == nil || c.Image == "" {

			ServersConfigs, err := getPredictorServerConfigs()

			if err != nil {
				seldondeploymentLog.Error(err, "Failed to read prepacked model servers from configmap")
			}

			_, ok := ServersConfigs[string(*pu.Implementation)]
			if !ok {
				allErrs = append(allErrs, field.Invalid(fldPath, pu.Name, "No entry in predictors map for "+string(*pu.Implementation)))
			}
		}
	}

	if pu.Logger != nil {
		if pu.Logger.Mode == "" {
			allErrs = append(allErrs, field.Invalid(fldPath, pu.Logger.Mode, "No logger mode specified"))
		}
	}

	for i := 0; i < len(pu.Children); i++ {
		allErrs = r.checkPredictiveUnits(&pu.Children[i], p, fldPath.Child("children").Index(i), allErrs)
	}

	return allErrs
}

func checkTraffic(spec *SeldonDeploymentSpec, fldPath *field.Path, allErrs field.ErrorList) field.ErrorList {
	var trafficSum int32 = 0
	var shadows int = 0
	for i := 0; i < len(spec.Predictors); i++ {
		p := spec.Predictors[i]

		if p.Shadow == true {
			shadows += 1
			if shadows > 1 {
				allErrs = append(allErrs, field.Invalid(fldPath, spec.Predictors[i].Name, "Multiple shadows are not allowed"))
			}
			if p.Traffic < 0 || p.Traffic > 100 {
				allErrs = append(allErrs, field.Invalid(fldPath, spec.Predictors[i].Name, "shadow traffic is illegal, the traffic number should be between [0, 100]"))
			}
		} else {
			trafficSum = trafficSum + p.Traffic
		}
	}

	if trafficSum != 100 && (len(spec.Predictors)-shadows) > 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.Predictors[0].Name, "Traffic must sum to 100 for multiple predictors"))
	}
	if trafficSum > 0 && trafficSum < 100 && len(spec.Predictors) == 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, spec.Predictors[0].Name, "Traffic must sum be 100 for a single predictor when set"))
	}

	return allErrs
}

func sizeOfGraph(p *PredictiveUnit) int {
	count := 0
	for _, child := range p.Children {
		count = count + sizeOfGraph(&child)
	}
	return count + 1
}

func collectTransports(pu *PredictiveUnit, transportsFound map[EndpointType]bool) {
	if pu.Endpoint != nil && pu.Endpoint.Type != "" {
		transportsFound[pu.Endpoint.Type] = true
	}
	for _, c := range pu.Children {
		collectTransports(&c, transportsFound)
	}
}

const (
	ENV_KAFKA_BROKER       = "KAFKA_BROKER"
	ENV_KAFKA_INPUT_TOPIC  = "KAFKA_INPUT_TOPIC"
	ENV_KAFKA_OUTPUT_TOPIC = "KAFKA_OUTPUT_TOPIC"
)

func (r *SeldonDeploymentSpec) validateSvcNameAnnotations(allErrs field.ErrorList) field.ErrorList {
	keys := make(map[string]bool)
	for i, p := range r.Predictors {
		if annotation, hasAnnotation := p.Annotations[ANNOTATION_CUSTOM_SVC_NAME]; hasAnnotation {
			if _, found := keys[annotation]; found {
				fldPath := field.NewPath("spec").Child("predictors").Index(i)
				allErrs = append(allErrs, field.Invalid(fldPath, p.Name, fmt.Sprintf("Found duplicate service name in %s with value %s", ANNOTATION_CUSTOM_SVC_NAME, annotation)))
			}
			keys[annotation] = true
		}
	}
	return allErrs
}

func (r *SeldonDeploymentSpec) validateKafka(allErrs field.ErrorList) field.ErrorList {
	if r.ServerType == ServerKafka {
		for i, p := range r.Predictors {
			if len(p.SvcOrchSpec.Env) == 0 {
				fldPath := field.NewPath("spec").Child("predictors").Index(i)
				allErrs = append(allErrs, field.Invalid(fldPath, p.Name, "For kafka please supply svcOrchSpec envs KAFKA_BROKER, KAFKA_INPUT_TOPIC, KAFKA_OUTPUT_TOPIC"))
			} else {
				found := 0
				for _, env := range p.SvcOrchSpec.Env {
					switch env.Name {
					case ENV_KAFKA_BROKER, ENV_KAFKA_INPUT_TOPIC, ENV_KAFKA_OUTPUT_TOPIC:
						found = found + 1
					}
				}
				if found < 3 {
					fldPath := field.NewPath("spec").Child("predictors").Index(i)
					allErrs = append(allErrs, field.Invalid(fldPath, p.Name, "For kafka please supply svcOrchSpec envs KAFKA_BROKER, KAFKA_INPUT_TOPIC, KAFKA_OUTPUT_TOPIC"))
				}
			}
		}
	}
	return allErrs
}

func (r *SeldonDeploymentSpec) validateShadow(allErrs field.ErrorList) field.ErrorList {
	if len(r.Predictors) == 1 && r.Predictors[0].Shadow {
		fldPath := field.NewPath("spec").Child("predictors").Index(0)
		allErrs = append(allErrs, field.Invalid(fldPath, r.Predictors[0].Name, "Shadow can not exist as only predictor"))
	}
	return allErrs
}

func (r *SeldonDeploymentSpec) ValidateSeldonDeployment() error {
	var allErrs field.ErrorList

	if r.Protocol != "" && !(r.Protocol == ProtocolSeldon || r.Protocol == ProtocolTensorflow || r.Protocol == ProtocolKFServing || r.Protocol == ProtocolV2) {
		fldPath := field.NewPath("spec")
		allErrs = append(allErrs, field.Invalid(fldPath, r.Protocol, "Invalid protocol"))
	}

	if r.Transport != "" && !(r.Transport == TransportRest || r.Transport == TransportGrpc) {
		fldPath := field.NewPath("spec")
		allErrs = append(allErrs, field.Invalid(fldPath, r.Transport, "Invalid transport"))
	}

	if r.ServerType != "" && !(r.ServerType == ServerRPC || r.ServerType == ServerKafka) {
		fldPath := field.NewPath("spec")
		allErrs = append(allErrs, field.Invalid(fldPath, r.ServerType, "Invalid serverType"))
	}

	allErrs = r.validateKafka(allErrs)
	allErrs = r.validateShadow(allErrs)
	allErrs = r.validateSvcNameAnnotations(allErrs)

	transports := make(map[EndpointType]bool)

	if len(r.Predictors) == 0 {
		fldPath := field.NewPath("spec")
		allErrs = append(allErrs, field.Invalid(fldPath, r.Transport, "Graph contains no predictors"))
	}

	predictorNames := make(map[string]bool)
	for i, p := range r.Predictors {

		collectTransports(&p.Graph, transports)

		_, noEngine := p.Annotations[ANNOTATION_NO_ENGINE]
		if noEngine && sizeOfGraph(&p.Graph) > 1 {
			fldPath := field.NewPath("spec").Child("predictors").Index(i)
			allErrs = append(allErrs, field.Invalid(fldPath, p.Name, "Running without engine only valid for single element graphs"))
		}

		if _, present := predictorNames[p.Name]; present {
			fldPath := field.NewPath("spec").Child("predictors").Index(i)
			allErrs = append(allErrs, field.Invalid(fldPath, p.Name, "Duplicate predictor name"))
		}
		predictorNames[p.Name] = true

		allErrs = r.checkPredictiveUnits(&p.Graph, &p, field.NewPath("spec").Child("predictors").Index(i).Child("graph"), allErrs)
	}

	if len(transports) > 1 {
		fldPath := field.NewPath("spec")
		allErrs = append(allErrs, field.Invalid(fldPath, "", "Multiple endpoint.types found - can only have 1 type in graph. Please use spec.transport"))
	} else if len(transports) == 1 && r.Transport != "" {
		for k := range transports {
			if (k == REST && r.Transport != TransportRest) || (k == GRPC && r.Transport != TransportGrpc) {
				fldPath := field.NewPath("spec")
				allErrs = append(allErrs, field.Invalid(fldPath, "", "Mixed transport types found. Remove graph endpoint.types if transport set at deployment level"))
			}
		}
	}

	allErrs = checkTraffic(r, field.NewPath("spec"), allErrs)

	if len(allErrs) == 0 {
		return nil
	}

	return apierrors.NewInvalid(
		schema.GroupKind{Group: "machinelearing.seldon.io", Kind: "SeldonDeployment"},
		r.Name, allErrs)

}

/// ---

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!

// TODO(user): change verbs to "verbs=create;update;delete" if you want to enable deletion validation.
// +kubebuilder:webhook:webhookVersions=v1,verbs=create;update,path=/validate-machinelearning-seldon-io-v1-seldondeployment,mutating=false,failurePolicy=fail,sideEffects=None,admissionReviewVersions=v1;v1beta1,groups=machinelearning.seldon.io,resources=seldondeployments,versions=v1,name=v1.vseldondeployment.kb.io

var _ webhook.Validator = &SeldonDeployment{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *SeldonDeployment) ValidateCreate() error {
	seldondeploymentLog.Info("Validating v1 Webhook called for CREATE", "name", r.Name)
	return r.Spec.ValidateSeldonDeployment()
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *SeldonDeployment) ValidateUpdate(old runtime.Object) error {
	seldondeploymentLog.Info("Validating v1 webhook called for UPDATE", "name", r.Name)
	return r.Spec.ValidateSeldonDeployment()
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *SeldonDeployment) ValidateDelete() error {
	seldondeploymentLog.Info("Validating v1 webhook called for DELETE", "name", r.Name)

	// TODO(user): fill in your validation logic upon object deletion.
	return nil
}
