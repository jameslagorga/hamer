IMAGE_NAME := gcr.io/lagorgeous-helping-hands/hamer:latest
DOCKERFILE := docker/hamer-dev.Dockerfile
DOCKER_BUILD_PLATFORM := linux/amd64 # Default to amd64 for GKE
RUN_CHART := hamer-demo.yaml

.PHONY: all build push

all: build push

build:
	docker build --platform $(DOCKER_BUILD_PLATFORM) -t $(IMAGE_NAME) -f $(DOCKERFILE) .

push:
	docker push $(IMAGE_NAME)

run:
	kubectl apply -f  $(RUN_CHART)
