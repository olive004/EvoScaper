# EvoScaper

Pre-print: Generative design of synthetic gene circuits for functional and evolutionary properties ([BioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.26.678595v1))

![Project overview](assets/cvae_figure1.jpg)

## Description

This repository contains all the code used to train neural networks on simulated genetic circuit dynamics and the notebooks used for analysis in the paper. The `src/evoscaper` repository contains scripts for simulating circuits, initialising the conditional variational autoencoder (CVAE) model, training batches of models, and verifying that generated circuits adhere to the function they were prompted with. 

![Ruggedness of two adaptable RNA circuits](assets/ys_sample.png)

## Usage

### Installation

The following installation instructions pertain to Linux machines, but can be adjusted for Windows with minimal changes.

A docker container is available for reproducing simulation and model training. For convenience, run the `docker/start_docker.sh` source file or run commands within this script. Once the container is up, run the `docker/post_install.sh` to ensure custom dependencies were installed correctly.

```bash
bash docker/start_docker.sh bash docker/post_install.sh
```

