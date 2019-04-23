FROM nvcr.io/nvidia/pytorch:19.03-py3

# PyTorch
RUN pip install -U torch torchvision

# Basic images processing
RUN pip install -U imageio numpy scipy opencv-python-headless scikit-image

# Graph
RUN pip install -U torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# QRNN
RUN pip install -U cupy
RUN pip install git+https://github.com/CorcovadoMing/pynvrtc.git

# Apex
RUN apt update && apt install -y curl
RUN curl https://raw.githubusercontent.com/apex/apex/master/install.sh | sh

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
