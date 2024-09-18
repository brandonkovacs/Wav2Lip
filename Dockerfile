FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN export DEBIAN_FRONTEND=noninteractive RUNLEVEL=1 ; \
     apt-get update && apt-get install -y --no-install-recommends \
          build-essential cmake git curl ca-certificates \
          vim wget \
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
     numpy=1.26.4 cudatoolkit librosa opencv tqdm numba ffmpeg fastapi python-multipart \
     -c pytorch -c conda-forge -c nvidia

# Upgrade pip and install remaining dependencies
RUN pip3 install --upgrade pip
RUN pip3 install opencv-contrib-python opencv-python python-ffmpeg mediapipe

# Download face detector model 
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN curl -SL -o /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

# Download Wav2Lip models
RUN wget 'https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip.pth?download=true' -O /app/checkpoints/wav2lip.pth
RUN wget 'https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth?download=true' -O /app/checkpoints/wav2lip_gan.pth

# Install Real-ESRGAN and dependencies
RUN git clone --recursive https://github.com/xinntao/Real-ESRGAN.git /app/Real-ESRGAN
WORKDIR /app/Real-ESRGAN
RUN pip3 install git+https://github.com/XPixelGroup/BasicSR.git
RUN pip3 install ffmpeg-python facexlib gfpgan gdown
RUN pip3 install -r /app/Real-ESRGAN/requirements.txt
RUN python3 /app/Real-ESRGAN/setup.py develop

# Download Real-ESRGAN Weights
RUN wget 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' -O /app/Real-ESRGAN/weights/RealESRGAN_x4plus.pth

# Download GFPGAN Weights
RUN mkdir -p /app/gfpgan/weights
RUN mkdir -p /opt/conda/lib/python3.10/site-packages/gfpgan/weights
RUN wget 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth' -O /app/gfpgan/weights/detection_Resnet50_Final.pth
RUN wget 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth' -O /app/gfpgan/weights/parsing_parsenet.pth
RUN wget 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth' -O /opt/conda/lib/python3.10/site-packages/gfpgan/weights/GFPGANv1.3.pth

WORKDIR /app

# Run as webservice
ENTRYPOINT ["fastapi", "run", "/app/api.py", "--port", "80"]

# Docker CLI mode
#ENTRYPOINT [ "python3", "/app/inference.py" ]

# Container CLI:
# python3 inference.py --checkpoint_path /checkpoints/wav2lip_gan.pth --face /workspace/video.mp4 --audio /workspace/audio.wav --outfile /workspace/output.mp4

# ESRGAN Upscale
# python inference_realesrgan_video.py -i /workspace/video.mp4 -o /workspace/ -n RealESRGAN_x4plus --face_enhance -s 1.5
