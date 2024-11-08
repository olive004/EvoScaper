# syntax=docker/dockerfile:1

# https://quay.io/repository/biocontainers/intarna
FROM quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0 AS intarna_build

# FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS workdir_build
# FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS workdir_build
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04 AS workdir_build

################################################################################
#  Install python dependencies
################################################################################
WORKDIR /usr/lib
COPY ./requirements.txt /usr/lib/install_requirements/requirements.txt
RUN apt-get update && \
    yes | apt-get install python3-pip
RUN yes | apt install git
RUN cd /usr/lib/install_requirements && \
    pip install -r ./requirements.txt 
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN python3 -m pipinstall "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/
# RUN pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

################################################################################
#  setup environment
################################################################################
ENV ROOT_DIR=/workdir
ENV PYTHONPATH=/workdir
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV MNT ./..
WORKDIR /workdir
# ENTRYPOINT ["/bin/bash"]

################################################################################
#  Copy IntaRNA dependencies
################################################################################
COPY --from=intarna_build /usr/local/bin/IntaRNA /usr/local/bin/
COPY --from=intarna_build /usr/local/include/IntaRNA /usr/local/include/
COPY --from=intarna_build /usr/local/lib/* /usr/local/lib/
# COPY --from=intarna_build /usr/lib/x86_64-linux-gnu/* /usr/lib/x86_64-linux-gnu/


################################################################################
#  Install conda and IntaRNA dependencies
################################################################################
# RUN yes | apt-get install wget && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN (sleep 1; echo yes; sleep 1; echo /root/miniconda3; sleep 1; echo yes) | bash Miniconda3-latest-Linux-x86_64.sh
# RUN . /root/miniconda3/bin/activate && \
#     yes | conda install -c conda-forge -c bioconda intarna

# # Switch up the IntaRNAs
# RUN mv /root/miniconda3/bin/IntaRNA /root/miniconda3/bin/IntaRNA2
# RUN mv /usr/local/bin/IntaRNA /root/miniconda3/bin/IntaRNA
# RUN ln -s /root/miniconda3/bin/IntaRNA /usr/local/bin/IntaRNA