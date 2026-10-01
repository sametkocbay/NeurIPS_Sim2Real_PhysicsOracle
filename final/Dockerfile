# PhysicsOracle, RealPDE Track 1: training package.
# The devel variant of the reference image: train() compiles the ensemble ahead of time (AOTInductor), which needs
# nvcc and a host C++ compiler (gcc 11.4) at build time of the graphs. CUDA 12.1, torch 2.2.2, numpy 1.26.4.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
RUN pip install --no-cache-dir h5py==3.11.0 pyyaml==6.0.1 numpy==1.26.4
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /workspace
COPY . /workspace
