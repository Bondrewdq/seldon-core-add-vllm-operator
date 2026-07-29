PROMETHEUS_STACK_VERSION ?= 87.21.0
DCGM_EXPORTER_VERSION ?= 3.3.1
NAMESPACE ?= monitoring

.PHONY: install status uninstall

install:
	kubectl apply -f namespace.yaml
	helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
		--version $(PROMETHEUS_STACK_VERSION) \
		--namespace $(NAMESPACE) \
		--values kube-prometheus-stack-values.yaml \
		--wait --timeout 15m
	kubectl apply -k .
	helm upgrade --install dcgm-exporter nvidia-dcgm/dcgm-exporter \
		--version $(DCGM_EXPORTER_VERSION) \
		--namespace $(NAMESPACE) \
		--values dcgm-exporter-values.yaml \
		--wait --timeout 10m

status:
	kubectl -n $(NAMESPACE) get pods
	kubectl -n $(NAMESPACE) get podmonitor,servicemonitor,prometheusrule

uninstall:
	-helm uninstall dcgm-exporter --namespace $(NAMESPACE)
	-kubectl delete -k . --ignore-not-found
	-helm uninstall kube-prometheus-stack --namespace $(NAMESPACE)
