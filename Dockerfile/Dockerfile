FROM nvcr.io/nvidia/pytorch:20.01-py3

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
RUN pip install torchtext spacy imageio numpy scipy opencv-python-headless scikit-image

# Graph
#RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
#RUN pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 -f https://s3.eu-central-1.amazonaws.com/pytorch-geometric.com/whl/torch-1.4.0.html
#RUN pip install torch-cluster torch-spline-conv torch-geometric

# QRNN
RUN pip install cupy
RUN pip install git+https://github.com/CorcovadoMing/pynvrtc.git

# RAPIDS
RUN conda update numba -y
RUN conda install -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.11 python=3.6 cudatoolkit=10.1

# Jupyter
RUN pip install jupyterlab ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# OpenAI gym
RUN apt update && apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb
RUN pip install lockfile glfw
RUN git clone https://github.com/openai/gym.git && cd gym && pip install -e .
RUN git clone https://github.com/benelot/pybullet-gym.git && cd pybullet-gym && pip install -e .
COPY run.sh /

# Visualization
RUN pip install graphviz hiddenlayer nvidia-ml-py3 seaborn pyro-ppl botorch gpytorch scikit-learn ax-platform pennylane pennylane-forest bindsnet adabound deepdish boxx pydicom --no-deps torch

# Julia
COPY julia-1.1.1 /opt/julia
RUN ln -s /opt/julia/bin/julia /bin
RUN julia -e 'using Pkg; Pkg.add("IJulia")'

# Tests
RUN conda remove -y wrapt
RUN pip install yapf pylint pytest texttable

# Augmentation and preprocessing
RUN pip install git+https://github.com/pytorch/contrib.git


# Fontend
#RUN conda remove -y PyYAML
RUN pip install psutil flask_compress pyyaml dash dash-daq dash-bootstrap-components pytz

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

RUN pip install pytorch-nlp pygsheets hyperopt tensorpack nevergrad[all]
RUN apt update && apt install -y sox libsox-dev libsox-fmt-all
RUN pip install http://download.pytorch.org/whl/torchaudio-0.2-cp36-cp36m-linux_x86_64.whl

RUN apt-get update && apt-get install -y build-essential libcap-dev cmake libopenmpi-dev zlib1g-dev libsdl2-gfx-dev libsdl2-ttf-dev
RUN pip install python-prctl tensorflow-gpu tensorflow-datasets gast
#RUN pip install -U numpy
#RUN pip install gast==0.2.2
#RUN pip install numpy==1.16.4

RUN pip install -U nibabel altair vega_datasets torchlars kymatio scikit-cuda cupy
RUN pip install git+git://github.com/deepmind/bsuite.git
RUN pip install tifffile pytorch-metric-learning prenlp cvxpylayers
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git
RUN pip install gym-retro pybullet procgen stable-baselines[mpi] gym-super-mario-bros
RUN git clone -b v2.0.4 https://github.com/google-research/football.git
RUN mkdir -p football/third_party/gfootball_engine/lib
RUN wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.0.4.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
RUN cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip install .
RUN apt install -y python-pygame && pip install pygame && pip install git+https://github.com/ntasfi/PyGame-Learning-Environment.git && pip install git+https://github.com/kenjyoung/MinAtar.git && pip install git+https://github.com/qlan3/gym-games.git
RUN apt install -y software-properties-common && apt-get update && add-apt-repository ppa:openjdk-r/ppa && apt-get update && apt-get install -y openjdk-8-jdk && pip install -U minerl
RUN apt install -y curl && curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && apt-get update && apt-get install bazel -y
RUN apt-get install -y libffi-dev gettext
RUN git clone https://github.com/deepmind/lab.git && cd lab && bazel build -c opt --python_version=PY3 //python/pip_package:build_pip_package && ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg && pip install /tmp/dmlab_pkg/deepmind_lab-1.0-py3-none-any.whl --force-reinstall
RUN pip install gym_deepmindlab
RUN apt-get install -y chromium-driver chromium-browser && pip install git+https://github.com/CorcovadoMing/gym-chrome-dino

# Unstaged change

# Version tag
ENV _RESEARCHLIB_IMAGE_TAG 20.02

