#!/bin/sh

pip install jax==0.4.29
pip install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install -U chex
pip install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@599990dcac56e7678f45269d2fc9df736d25f356#egg=bioreaction
pip install git+https://github.com/olive004/synbio_morpher.git@bc7aaf284fcf5b10abf591f5f2cf6c898f45861f#egg=synbio_morpher
# pip install -e src/bioreaction
