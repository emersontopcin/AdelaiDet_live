#teset
# Nome da imagem Docker
IMAGE_NAME := adelaidet_live_image
# Nome do container Docker
CONTAINER_NAME := adelaidet_live_container

.PHONY: docker-build docker-run docker-start docker-stop docker-remove-container docker-remove-image docker-exec

# Regra para construir a imagem Docker
docker-build:
	docker build -t $(IMAGE_NAME) .

# Regra para executar o container Docker
docker-run:
	xhost +local:root
		docker run -it \
		--gpus all \
		--name $(CONTAINER_NAME) \
		--privileged \
		-e DISPLAY=$(DISPLAY) \
		--network host \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /media/live-segcom/nuscenes:/home/nuscenes \
		$(IMAGE_NAME)

# Regra para iniciar o container Docker
docker-start:
	docker start $(CONTAINER_NAME)

# Regra para parar o container Docker
docker-stop:
	docker stop $(CONTAINER_NAME)

# Regra para remover o container Docker
docker-remove-container:
	docker rm $(CONTAINER_NAME)

# Regra para remover a imagem
docker-remove-image:
	docker rmi $(IMAGE_NAME)

# Regra para executar um comando no container Docker
docker-exec:
	$(MAKE) docker-start
	xhost +si:localuser:root
	docker exec -it $(CONTAINER_NAME) /bin/bash