#!/bin/sh

docker tag rf37535/researchlib:latest nvcr.io/nvidian/sae/researchlib:latest
docker push nvcr.io/nvidian/sae/researchlib:latest
