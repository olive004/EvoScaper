
# Reload nvidia info
sudo nvidia-container-cli --load-kmods info

# A pre-req for using gpus here is the NVIDIA Docker Container Toolkit

sudo docker pull quay.io/biocontainers/intarna:3.4.1--pl5321hdcf5f25_0
# sudo docker run --rm -it --entrypoint bash quay.io/biocontainers/intarna:3.4.1--pl5321hdcf5f25_0
# sudo docker pull nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
# sudo docker pull nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# If image not built yet
if [ "$(id -un)" != "wadh6511" ]; then
    sudo docker pull nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04
    source_directory="docker_mufasa"
else
    sudo docker pull nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
    source_directory="docker"
fi

sudo docker build -t genetic_glitch:latest $source_directory

cp ./requirements.txt ./$source_directory


# sudo docker create -it \
# --rm \
# --gpus all \
# --device /dev/nvidia0:/dev/nvidia0 \
# --device /dev/nvidia-modeset:/dev/nvidia-modeset \
# --device /dev/nvidia-caps:/dev/nvidia-caps \
# --device /dev/nvidia-uvm:/dev/nvidia-uvm \
# --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools \
# --device /dev/nvidiactl:/dev/nvidiactl \
# --name evo \
# --mount type=bind,source="$(pwd)",target=/workdir \
# genetic_glitch:latest
sudo docker create -it \
--rm \
--gpus all \
--name evo \
--runtime nvidia \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
sudo docker container start evo

# sudo docker exec -it evo cd /usr/lib/install_requirements && pip install -r ./requirements.txt
# sudo docker exec -it evo pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
sudo docker exec -it evo bash docker/post_install.sh
sudo docker exec -it evo /bin/bash 
# sudo docker container stop evo
