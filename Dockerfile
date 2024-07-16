FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN export DEBIAN_FRONTEND=noninteractive RUNLEVEL=1 ; \
     apt-get update && apt-get install -y --no-install-recommends \
          build-essential cmake git curl ca-certificates \
          vim \
          python3-pip python3-dev python3-wheel \
          libglib2.0-0 libxrender1 python3-soundfile \
          ffmpeg && \
	rm -rf /var/lib/apt/lists/* && \
     pip3 install --upgrade setuptools

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
      chmod +x ~/miniconda.sh && \
      ~/miniconda.sh -b -p /opt/conda && \
      rm ~/miniconda.sh

# Add conda to path
ENV PATH /opt/conda/bin:$PATH

# Copy wav2lip to /app
COPY . /app

# Install base conda environment with cuda support
RUN conda config --set always_yes yes --set changeps1 no && conda update -q conda
RUN conda install python=3.10 pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 \
     numpy=1.26.4 cudatoolkit librosa opencv tqdm numba ffmpeg -c pytorch -c conda-forge -c nvidia

# Upgrade pip and install remaining dependencies
RUN pip3 install --upgrade pip
RUN pip3 install opencv-contrib-python opencv-python python-ffmpeg mediapipe

# Download face detector model 
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN curl -SL -o /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

# Create working directory (to be mounted by docker)
RUN mkdir /workspace

WORKDIR /app
# python3 inference.py --checkpoint_path /checkpoints/wav2lip_gan.pth --face /workspace/video.mp4 --audio /workspace/audio.wav --outfile /workspace/output.mp4