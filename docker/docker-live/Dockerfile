FROM pytorch/pytorch:2.5.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y \
        tzdata \
        libglib2.0-0 \
        libgl1 \
        wget \
        htop \
        byobu \ 
        git \
        gcc \
        g++ \
        vim \
        libsm6 \
        libxext6 \
        libxrender-dev \
        lsb-core \
    && apt-get clean

WORKDIR /root/code

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install -e detectron2

# Install adelaiDet
RUN git clone https://github.com/emersontopcin/AdelaiDet_live.git adet
WORKDIR /root/code/adet
RUN pip install -e .

# Install python dependencies
RUN pip install nuscenes-devkit opencv-python

CMD ["bash"]