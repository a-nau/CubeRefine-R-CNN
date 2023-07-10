FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# See https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get -y update

# see https://serverfault.com/a/992421
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

RUN apt-get install -y libopencv-dev python3-opencv git cmake

RUN pip install -U pip

# General
RUN pip install scikit-image==0.18.* matplotlib imageio==2.31.1 opencv-python==4.8.0.74 scipy==1.10.1 pandas==2.0.3 joblib==1.3.1 einops==0.6.1
# Detectron2
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.11/index.html
RUN pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"
# Pytorch3D
RUN pip install fvcore==0.1.5.post20221221 
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
# PyTorch Geometric
RUN pip install pyg_lib==0.2.0 torch_scatter==2.0.9 torch_sparse==0.6.13 torch_cluster==1.6.0 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install torch_geometric==2.3.1
