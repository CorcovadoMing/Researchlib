FROM nvcr.io/nvidia/pytorch:19.05-py3

# PyTorch
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

# Basic images processing
RUN pip install -U imageio numpy scipy opencv-python-headless scikit-image

# Graph
RUN pip install -U torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# QRNN
RUN pip install -U cupy
RUN pip install git+https://github.com/CorcovadoMing/pynvrtc.git

# Apex
#RUN git clone https://github.com/NVIDIA/apex apex_build
#RUN cd apex_build && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# DALI
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

# RAPIDS
RUN pip install -U cudf-cuda100
RUN apt update && apt install -y libopenblas-base libomp-dev
RUN pip install -U cuml-cuda100
RUN pip install -U cugraph-cuda100
RUN pip install -U nvstrings-cuda100

# Jupyter
RUN pip install jupyterlab
RUN pip install ipywidgets
COPY jupyter /root/.jupyter
RUN jupyter nbextension enable --py widgetsnbextension
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash -
RUN apt install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager

# OpenAI gym
RUN apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb
RUN pip install -U lockfile
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
RUN pip install -U graphviz hiddenlayer nvidia-ml-py3
RUN pip install -U bqplot
RUN jupyter labextension install bqplot
