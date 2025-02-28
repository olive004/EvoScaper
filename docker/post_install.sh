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


pip3 install -r requirements.txt
pip3 install jax==0.4.29
pip3 install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip3 install -U chex
# pip3 install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@f903c39872de43e28b56653efda689bb082cb592#egg=bioreaction
pip3 install git+https://github.com/olive004/synbio_morpher.git@590dd3b477f1f57f1f448e59e93f677006d2e604#egg=synbio_morpher

if [ -d "src/core-bioreaction-simulation" ]; then
    echo "Directory src/bioreaction exists."
else
    cd src
    git clone https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git
    cd ..
fi
pip3 install -e src/core-bioreaction-simulation
pip3 install -e .
