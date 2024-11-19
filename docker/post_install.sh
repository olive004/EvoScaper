#!/bin/sh


# Install python3.11
if python3 -c 'import sys; exit(not (sys.version_info >= (3, 10)))'; then
    echo "Python version is greater than or equal to 3.10"
else
    echo "Python version is less than 3.10"
    apt update
    apt autoremove
    apt upgrade
    apt install software-properties-common -y
    add-apt-repository ppa:deadsnakes/ppa
    apt update
    apt install python3.11
    apt install python3.11-venv
    apt install python3.11-dev
    ls -la /usr/bin/python3
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
# rm /usr/bin/python3
# ln -s python3.11 /usr/bin/python3
# apt install curl
# curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
# python3 --version
# python3.11 -m pip install ipython
fi


pip install -r requirements.txt
pip install jax==0.4.29
pip install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install -U chex
# pip install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@599990dcac56e7678f45269d2fc9df736d25f356#egg=bioreaction
# pip install git+https://github.com/olive004/synbio_morpher.git@bc7aaf284fcf5b10abf591f5f2cf6c898f45861f#egg=synbio_morpher
# pip install -e src/bioreaction

if [ -d "src/bioreaction" ]; then
    echo "Directory src/bioreaction exists."
else
    cd src
    git clone https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git
    rm src/bioreaction
    cd ..
fi
pip install -e core-bioreaction-simulation/src/bioreaction