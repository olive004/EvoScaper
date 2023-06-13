
import os
from setuptools import find_packages, setup

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(CURRENT_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    

    
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
 
# https://www.tensorflow.org/install/pip#linux_setup
# conda install -c conda-forge cudatoolkit=11.8.0
# python3 -m pip install nvidia-cudnn-cu11==8.9.2.26 tensorflow==2.12.*
# mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
 
# activate env
# python3 -m pip install nvidia-cudnn-cu11==8.9.2.26 tensorflow==2.12.*
# echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $VIRTUAL_ENV/bin/activate
# echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib/:$CUDNN_PATH/lib' >> $VIRTUAL_ENV/bin/activate

setup(
    author="Olivia Gallup",
    author_email="olivia.gallup@gmail.com",
    classifiers=[
        # "Development Status :: 4 - Beta",
        # "Environment :: Console",
        "Intended Audience :: Science/Research",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        # "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_releases.html",
    ],
    description="EvoScaper: Predicting the robustness of genetic "
    "circuits against mutations",
    install_requires=[
        "absl-py>=1.0.0",
        "jax>=0.3.25",
        "jaxlib>=0.3.25",
        # "dm-haiku>=0.0.9",
        "numpy>=1.23.5",
        # "boto3>=1.24.28",
        # "typing_extensions>=3.10.0",
        # "joblib>=1.2.0",
        "tqdm>=4.56.0",
        # "regex>=2022.1.18",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Synthetic Biology", "Biotech", "Deep Learning", "JAX"],
    name='evoscaper',
    packages=find_packages(include=['evoscaper', 'evoscaper.*']),
    python_requires=">=3.8",
    tests_require=['pytest'],
    url="https://github.com/olive004/EvoScaper",
    version="0.0.1",
)
