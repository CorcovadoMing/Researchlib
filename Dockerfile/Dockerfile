FROM nvcr.io/nvidia/pytorch:19.03-py3

# PyTorch
RUN pip install -U torch torchvision

# Basic images processing
RUN pip install -U imageio numpy scipy opencv-python-headless

# Graph
RUN pip install -U torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# QRNN
RUN pip install -U cupy
RUN pip install git+https://github.com/CorcovadoMing/pynvrtc.git

# Apex
RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# DALI
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

# Jupyter
RUN pip install jupyterlab
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
COPY jupyter /root/.jupyter
