# Use the official PyTorch image with CUDA 12.6 as the base image
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

# Set environment variables for NVIDIA GPU access
# You may need to adjust these based on your host system's configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y git

# **FIXED STEP:** Install the necessary libraries for OpenCV
# libGL.so.1 is in libgl1-mesa-glx
# libgthread-2.0.so.0 is in libglib2.0-0
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Add any other build steps here, for example:
COPY . /sam3
WORKDIR /sam3
RUN pip install -e .
RUN pip install -e ".[train,dev]"
RUN pip install -e ".[notebooks]"
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install opencv-python einops

# RUN pip install -r requirements.txt