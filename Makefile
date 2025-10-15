IMAGE_NAME := gcr.io/lagorgeous-helping-hands/hamer:latest
DOCKERFILE := docker/hamer-dev.Dockerfile
DOCKER_BUILD_PLATFORM := linux/amd64 # Default to amd64 for GKE

.PHONY: all build push

all: build push

build:
	docker build --platform $(DOCKER_BUILD_PLATFORM) -t $(IMAGE_NAME) -f $(DOCKERFILE) .

push:
	docker push $(IMAGE_NAME)
