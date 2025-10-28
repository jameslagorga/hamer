IMAGE_NAME := gcr.io/lagorgeous-helping-hands/hamer:latest
DOCKERFILE := docker/hamer-dev.Dockerfile
DOCKER_BUILD_PLATFORM := linux/amd64 # Default to amd64 for GKE
RUN_CHART := hamer-demo.yaml

.PHONY: all build push demo render infer clear-subscription apply-subscriber delete-subscriber apply-hand-counter delete-hand-counter clear-hand-counter-subscription

all: build push

build:
	docker build --platform $(DOCKER_BUILD_PLATFORM) -t $(IMAGE_NAME) -f docker/hamer-dev.Dockerfile .

push:
	docker push $(IMAGE_NAME)

demo:
	kubectl apply -f hamer-demo.yaml

render:
	kubectl apply -f hamer-render.yaml

infer:
	kubectl apply -f hamer-infer.yaml

apply-subscriber:
	kubectl apply -f hamer-subscriber-deployment.yaml

delete-subscriber:
	kubectl delete -f hamer-subscriber-deployment.yaml --ignore-not-found=true

clear-subscription:
	gcloud pubsub subscriptions delete hamer-subscription --quiet || true
	gcloud pubsub subscriptions create hamer-subscription --topic=frame-processing-topic

apply-hand-counter:
	kubectl apply -f hamer-hand-counter-deployment.yaml

delete-hand-counter:
	kubectl delete -f hamer-hand-counter-deployment.yaml --ignore-not-found=true

clear-hand-counter-subscription:
	gcloud pubsub subscriptions delete hamer-hand-counter-subscription --quiet || true
	gcloud pubsub subscriptions create hamer-hand-counter-subscription --topic=frame-processing-topic
