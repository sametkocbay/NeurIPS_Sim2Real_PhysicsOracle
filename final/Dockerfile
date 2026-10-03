# PhysicsOracle: training package.
# The development image provides CUDA and C++ tools for AOTInductor compilation.
FROM --platform=linux/amd64 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN pip install --no-cache-dir h5py==3.11.0 pyyaml==6.0.1 numpy==1.26.4

ENV CUDA_HOME=/usr/local/cuda
WORKDIR /workspace
COPY . /workspace
