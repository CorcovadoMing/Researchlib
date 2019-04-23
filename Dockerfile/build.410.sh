#!/bin/sh

docker build -t rf37535/researchlib:410 -f Dockerfile.410 .
docker push rf37535/researchlib:410
