FROM nvcr.io/nvidia/tensorrt:19.06-py3

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_SERVER_BRANCH=master

RUN apt-get update &&\
    apt-get install -y sudo git bash

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/opt/miniconda/bin:${PATH}


# Prepare onnxruntime repository & build onnxruntime with TensorRT
COPY onnxruntime ./onnxruntime
RUN /bin/sh onnxruntime/dockerfiles/install_common_deps.sh &&\
    cp onnxruntime/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /workspace/tensorrt --config Release --build_wheel --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) &&\
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd .. &&\
    rm -rf onnxruntime cmake-3.14.3-Linux-x86_64.tar.gz cmake-3.14.3-Linux-x86_64

# PyTorch
RUN pip install -U torch torchvision
RUN pip install -U torchtext spacy

# Basic images processing
RUN pip install -U imageio numpy scipy opencv-python-headless scikit-image

# Graph
RUN pip install -U torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# QRNN
RUN pip install -U cupy
RUN pip install git+https://github.com/CorcovadoMing/pynvrtc.git

# Apex
# RUN pip install -U pip
# RUN git clone https://github.com/NVIDIA/apex apex_build
# RUN cd apex_build && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# DALI
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

# RAPIDS
RUN pip install -U cudf-cuda100
RUN apt update && apt install -y libopenblas-base libomp-dev
RUN pip install -U cuml-cuda100
RUN pip install -U cugraph-cuda100
RUN pip install -U nvstrings-cuda100

# Jupyter
RUN pip install -U jupyterlab
RUN pip install -U ipywidgets
COPY jupyter /root/.jupyter
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# OpenAI gym
RUN apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb
RUN pip install -U lockfile glfw
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
RUN pip install -U graphviz hiddenlayer nvidia-ml-py3 seaborn

# Probability PyTorch
RUN pip install -U pyro-ppl botorch gpytorch ax-platform

# QC (Ignore for now)
# RUN pip install -U pennylane pennylane-forest bindsnet

# MISC
RUN pip install -U adabound
RUN pip install -U deepdish boxx 

RUN pip install -U pydicom

# Julia
COPY julia-1.1.1 /opt/julia
RUN ln -s /opt/julia/bin/julia /bin
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

# Tests
RUN pip install -U yapf pylint pytest
RUN pip install -U zarr

# Augmentation and preprocessing
RUN pip install -U albumentations
RUN pip install git+https://github.com/pytorch/contrib.git

RUN pip install -U texttable

# Version tag
ENV _RESEARCHLIB_IMAGE_TAG 19.08