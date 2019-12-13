FROM nvcr.io/nvidia/pytorch:19.10-py3

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_SERVER_BRANCH=master

RUN apt-get update &&\
    apt-get install -y sudo git bash

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/opt/miniconda/bin:${PATH}


# Prepare onnxruntime repository & build onnxruntime with TensorRT
COPY onnxruntime ./onnxruntime
COPY install_common_deps.sh ./install_common_deps.sh
RUN /bin/sh ./install_common_deps.sh
RUN cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
    cp onnxruntime/ThirdPartyNotices.txt /code/ThirdPartyNotices.txt &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /usr/src/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime cmake-3.14.3-Linux-x86_64.tar.gz cmake-3.14.3-Linux-x86_64 install_common_deps.sh

# Text
RUN pip install torchtext spacy 

# Basic images processing
RUN pip install imageio numpy scipy opencv-python-headless scikit-image

# Graph
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# QRNN
RUN pip install cupy
RUN pip install git+https://github.com/CorcovadoMing/pynvrtc.git

# RAPIDS
RUN conda install -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.10 python=3.6 cudatoolkit=10.1

# Jupyter
RUN pip install jupyterlab
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# OpenAI gym
RUN apt update && apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb
RUN pip install lockfile glfw
RUN mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    wget https://www.roboti.us/download/mjpro150_linux.zip  && \
    unzip mjpro150_linux.zip && \
    wget https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    mv mujoco200_linux mujoco200
ARG MUJOCO_KEY
ENV MUJOCO_KEY=$MUJOCO_KEY
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
RUN echo $MUJOCO_KEY | base64 --decode > /root/.mujoco/mjkey.txt
RUN pip install 'gym[all]'
COPY run.sh /

# Visualization
RUN pip install graphviz hiddenlayer nvidia-ml-py3 seaborn

# Probability PyTorch
RUN pip install pyro-ppl botorch gpytorch scikit-learn ax-platform --no-deps torch

# QC
RUN pip install pennylane pennylane-forest bindsnet --no-deps torch

# MISC
RUN pip install adabound
RUN pip install deepdish boxx 

RUN pip install pydicom

# Julia
COPY julia-1.1.1 /opt/julia
RUN ln -s /opt/julia/bin/julia /bin
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

# Tests
RUN conda remove -y wrapt
RUN pip install yapf pylint pytest

# Augmentation and preprocessing
RUN pip install albumentations
RUN pip install git+https://github.com/pytorch/contrib.git
RUN pip install texttable


# Fontend
#RUN conda remove -y PyYAML
RUN pip install flask_compress pyyaml dash dash-daq dash-bootstrap-components pytz
RUN pip install psutil

# Adaptive robust loss
RUN pip install git+https://github.com/jonbarron/robust_loss_pytorch

# Install Redis.
RUN \
  cd /tmp && \
  wget http://download.redis.io/redis-stable.tar.gz && \
  tar xvzf redis-stable.tar.gz && \
  cd redis-stable && \
  make && \
  make install && \
  cp -f src/redis-sentinel /usr/local/bin && \
  mkdir -p /etc/redis && \
  cp -f *.conf /etc/redis && \
  rm -rf /tmp/redis-stable* && \
  sed -i 's/^\(bind .*\)$/# \1/' /etc/redis/redis.conf && \
  sed -i 's/^\(daemonize .*\)$/# \1/' /etc/redis/redis.conf && \
  sed -i 's/^\(dir .*\)$/# \1\ndir \/data/' /etc/redis/redis.conf && \
  sed -i 's/^\(logfile .*\)$/# \1/' /etc/redis/redis.conf

RUN pip install redis
VOLUME ["/data"]

RUN pip install pytorch-nlp pygsheets
RUN apt update && apt install -y sox libsox-dev libsox-fmt-all
RUN pip install http://download.pytorch.org/whl/torchaudio-0.2-cp36-cp36m-linux_x86_64.whl
RUN pip install hyperopt

RUN pip install tensorpack
RUN apt-get update && apt-get install -y build-essential libcap-dev
RUN pip install python-prctl

RUN pip install tensorflow-gpu
RUN pip install tensorflow-datasets
RUN pip install -U numpy
RUN pip install gast==0.2.2
RUN pip install numpy==1.16.4

RUN pip install -U nibabel
RUN pip install git+https://github.com/CorcovadoMing/box-convolutions.git

# Version tag
ENV _RESEARCHLIB_IMAGE_TAG 19.11.0

