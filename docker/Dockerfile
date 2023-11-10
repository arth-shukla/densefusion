# Adapted from https://github.com/haosulab/ManiSkill2/blob/main/docker/Dockerfile

# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.4.1 as micromamba

# the actual base image
FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh


USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

# Optional: you can now populate a conda environment:
RUN micromamba install --yes --name base --channel conda-forge \
      jq python=3.9 setuptools= && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

USER root

RUN mkdir /work
WORKDIR /work

ARG CACHE_DATE=not_a_date

# installations for calql
RUN ln -s /usr/lib/nvidia /usr/lib/nvidia-430 && mkdir -p ~/.mujoco/mujoco200 && wget https://www.roboti.us/download/mujoco200_linux.zip && unzip mujoco200_linux.zip -d ~/.mujoco/ && rm mujoco200_linux.zip && mkdir -p ~/.mujoco/mujoco200 && mv ~/.mujoco/mujoco200_linux/* ~/.mujoco/mujoco200
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/ && rm mujoco210-linux-x86_64.tar.gz
RUN apt-get update && apt-get -y install patchelf

# package installs
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install "gymnasium==0.29.1" && \
    pip install "gymnax==0.0.6" && \
    pip install "chex==0.1.84" && \
    pip install "distrax==0.1.4" && \
    pip install "flax==0.7.5" && \
    pip install "optax==0.1.7" && \
    pip install "moviepy==1.0.3" && \
    pip install "tensorboard==2.15.0" && \
    pip install "tensorboardX==2.6.2.2" && \
    pip install "wandb==0.15.12" && \
    pip install "omegaconf==2.3.0" && \
    pip install "opencv-python==4.8.1.78" && \
    pip install "transforms3d==0.4.1" && \
    pip install "tqdm==4.66.1" && \pip install "dacite==1.8.1" && \
    pip install "matplotlib==3.8.0" && \
    pip install "ml_collections==0.1.1" && \
    pip install "h5py==3.10.0" && \
    pip install "gymnasium_robotics==1.2.3" && \
    pip install "mani_skill2==0.5.2"
