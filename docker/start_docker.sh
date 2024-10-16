
# Reload nvidia info
sudo nvidia-container-cli --load-kmods info

# A pre-req for using gpus here is the NVIDIA Docker Container Toolkit

sudo docker pull quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0
# sudo docker run --rm -it --entrypoint bash quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0
# sudo docker pull nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
sudo docker pull nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# If image not built yet
sudo docker build -t genetic_glitch:latest docker

cp ./requirements.txt ./docker

sudo docker create -it \
--rm \
--gpus all \
--device /dev/nvidia0:/dev/nvidia0 \
--device /dev/nvidia-modeset:/dev/nvidia-modeset \
--device /dev/nvidia-caps:/dev/nvidia-caps \
--device /dev/nvidia-uvm:/dev/nvidia-uvm \
--device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
--device /dev/nvidiactl:/dev/nvidiactl \
--name gcg \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
sudo docker container start gcg
sudo docker exec -it gcg /bin/bash 
# sudo docker container stop gcg
