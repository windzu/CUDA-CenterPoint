# 1. Choose Base Image
# CUDA12.2 TensorRT8.6 CUDNN8.9 Ubuntu22.04
# FROM nvcr.io/nvidia/tensorrt:23.10-py3 

# CUDA12.6 TensorRT10.3 CUDNN9.3 Ubuntu22.04
FROM nvcr.io/nvidia/tensorrt:24.08-py3 

# Set environment variable to prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# 2. Change apt sources to USTC mirrors
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list

# 3. Install Basic Dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    locales \
    sudo \
    build-essential \
    git \
    vim \
    bash-completion \
    clang-format \
    cpplint \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# create non-root user
ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=vscode

RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME

RUN echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir -p /workspace && chown $USER_UID:$USER_GID /workspace

USER $USERNAME
ENV HOME=/home/$USERNAME
WORKDIR /workspace
