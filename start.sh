#!/bin/sh

docker run --runtime nvidia --rm -itd -v `pwd`:/source -w /source --net host --name researchlib-jupyter -e CUDA_VISIBLE_DEVICES=6,7 rf37535/researchlib jupyter lab --ip 0.0.0.0 --port 8899 --allow-root --no-browser

