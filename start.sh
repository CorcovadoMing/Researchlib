#!/bin/sh

docker run --runtime nvidia --rm -itd -v `pwd`:/source -w /source --net host --name researchlib-jupyter --ipc host -e CUDA_VISIBLE_DEVICES=$1 rf37535/researchlib:$2 jupyter lab --ip 0.0.0.0 --port 8899 --allow-root --no-browser

