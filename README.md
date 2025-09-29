# EvoScaper

Pre-print: Generative design of synthetic gene circuits for functional and evolutionary properties ([BioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.26.678595v1))

![Project overview](assets/cvae_figure1.jpg)

## Description

This repository contains all the code used to train neural networks on simulated genetic circuit dynamics and the notebooks used for analysis in the paper. The `src/evoscaper` repository contains scripts for simulating circuits, initialising the conditional variational autoencoder (CVAE) model, training batches of models, and verifying that generated circuits adhere to the function they were prompted with. 

![Ruggedness of two adaptable RNA circuits](assets/ys_sample.png)

## Usage

### Installation

The following installation instructions pertain to Linux machines, but can be adjusted for Windows with minimal changes.

A docker container is available for reproducing simulation and model training. For convenience, run the `docker/start_docker.sh` source file or run commands within this script. Once the container is up, run the `docker/post_install.sh` to ensure custom dependencies were installed correctly.

```bash
bash docker/start_docker.sh 
bash docker/post_install.sh
```

### Notebooks

As a starting point, a model can be trained with the first notebook `notebooks/01_cvae.ipynb`. Subsequently, model outputs can be verified with the second notebook `notebooks/02_cvae_verify.ipynb`. 

The remaining notebooks explore aspects of model parameters, training, Monte Carlo sampling simulations, and other visualisation / verification results.

### CLI

To train and verify multiple CVAE models with different configurations, modify the `src/evoscaper/run/cvae_multi.py` script with path names indicating configuration files. 

To calculate the ruggedness of a set of genetic circuits, run the `src/evoscaper/run/ruggedness.py` script with modifications to the config in the script, specifying parameters such as the number of mutated circuits and the mutation strategy. 
