# syntax=docker/dockerfile:1.7
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ARG ROCM_ARCH=gfx1151

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    cmake \
    ninja-build \
    build-essential \
    libeigen3-dev \
    libopencv-dev \
    libsqlite3-dev \
    libboost-all-dev \
    libceres-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgtest-dev \
    libgmock-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libcurl4-openssl-dev \
    libopenimageio-dev \
    openimageio-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG CMAKE_EXTRA_ARGS=""
ARG KEEP_SOURCE=0

COPY . /opt/colmap_src
RUN mkdir -p /opt/colmap_src/build \
    && cd /opt/colmap_src/build \
    && cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=OFF \
        -DHIP_ENABLED=ON \
        -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} \
        ${CMAKE_EXTRA_ARGS} \
    && ninja -j "$(nproc)" \
    && ninja install \
    && if [ "${KEEP_SOURCE}" = "0" ]; then rm -rf /opt/colmap_src; fi

WORKDIR /workspace
ENTRYPOINT ["colmap"]
