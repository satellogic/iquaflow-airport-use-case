PROJ_NAME=iquaflow-airplane-yolov5-use-case
CONTAINER_NAME="${PROJ_NAME}-${USER}"


ifndef DS_VOLUME
	DS_VOLUME=/Nas
endif

ifndef NB_PORT
	NB_PORT=8888
endif

ifndef MLF_PORT
	MLF_PORT=5000
endif

help:
	@echo "build -- builds the docker image"
	@echo "dockershell -- raises an interactive shell docker"
	@echo "notebookshell -- launches a notebook server"
	@echo "mlflow -- launches an mlflow server"
	@echo "downloads course materials"

build:
	docker build -t $(PROJ_NAME) .
	chmod 775 ./download.sh
	./download.sh

dockershell:
	docker run --rm --name $(CONTAINER_NAME) --gpus all \
	-p 9197:9197 \
	-v $(shell pwd):/iqf -v $(DS_VOLUME):/ds_volume \
	-it $(PROJ_NAME)

notebookshell:
	docker run --gpus all --privileged -itd --rm --name $(CONTAINER_NAME)-nb \
	-p ${NB_PORT}:${NB_PORT} \
	-v $(shell pwd):/iqf -v $(DS_VOLUME):/ds_volume \
	$(PROJ_NAME) \
	jupyter lab \
	--NotebookApp.token='IQF' \
	--no-browser \
	--ip=0.0.0.0 \
	--allow-root \
	--port=${NB_PORT}

mlflow:
	docker run --privileged -itd --rm --name $(CONTAINER_NAME)-mlf \
	-p ${MLF_PORT}:${MLF_PORT} \
	-v $(shell pwd):/iqf -v $(DS_VOLUME):/ds_volume \
	$(PROJ_NAME) \
	mlflow ui --host 0.0.0.0:$(MLF_PORT)

course:
	wget https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/datasets/course.zip
	chmod 775 ./course.zip && unzip -o ./course.zip && rm ./course.zip
