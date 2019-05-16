#!/bin/sh

docker run --runtime nvidia -itd --restart=always -v `pwd`:/source -w /source --net host --name researchlib-jupyter --ipc host -e CUDA_VISIBLE_DEVICES=$1 rf37535/researchlib:$2 sh /run.sh

