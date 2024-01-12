FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_SAMPLES_INC /smplify-x/torch-mesh-isect/cuda-samples/Common

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_INSTALL="apt-get install -y --no-install-recommends"
ARG PIP_INSTALL="python -m pip --no-cache-dir install --upgrade --default-timeout 100 -i https://pypi.tuna.tsinghua.edu.cn/simple"
ARG CONDA_INSTALL="conda install -y"
ARG WGET="wget --no-verbose --show-progress --progress=bar:force:noscroll"
ARG GIT_CLONE="git clone --depth 1"

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get -y upgrade && \
    # ==================================================================
    # tools & env
    # ------------------------------------------------------------------
    $APT_INSTALL \
        cmake \
        curl \
        git \
        sudo \
        vim \
        wget \
        zip \
        libboost-dev \
        libglib2.0-dev \
        libsm6 libxext6 libxrender-dev libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 freeglut3-dev \
        && \
    # chmod -R 777 /opt && \
    rm -rf /opt/conda && \
    $WGET https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    $CONDA_INSTALL python=3.8 && \
    $PIP_INSTALL pip setuptools && \
    $CONDA_INSTALL pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia && \
    # ==================================================================
    # code & models
    # ------------------------------------------------------------------
    $GIT_CLONE --recursive https://github.com/jasongzy/smplify-x.git && cd /smplify-x && \
    $PIP_INSTALL -r requirements.txt && \
    # git submodule update --init && cd torch-mesh-isect/cuda-samples && git init && git sparse-checkout init && git remote add origin https://github.com/NVIDIA/cuda-samples.git && git sparse-checkout set "Common" && git fetch --depth 1 origin && git checkout master && cd ../.. && \
    cd torch-mesh-isect && python setup.py install && cd .. && \
    $WGET https://ghproxy.net/https://github.com/jasongzy/smplify-x/releases/download/v1.0/models.tar.gz -O /tmp/models.tar.gz && \
    $WGET https://ghproxy.net/https://github.com/jasongzy/smplify-x/releases/download/v1.0/vposer.tar.gz -O /tmp/vposer.tar.gz && \
    tar -xf /tmp/models.tar.gz && \
    tar -xf /tmp/vposer.tar.gz && \
    # ==================================================================
    # config & cleanup
    # ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    conda clean -y --all && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/pip

WORKDIR /smplify-x
