#!/bin/bash
DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends \
        wget \
        zip \
        ca-certificates \
        build-essential \
        curl \
        libcurl4-openssl-dev \
        libssl-dev \

sudo wget --quiet https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz
tar zxf cmake-3.14.3-Linux-x86_64.tar.gz
rm -rf cmake-3.14.3-Linux-x86_64.tar.gz
