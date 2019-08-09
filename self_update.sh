#!/bin/sh

# Upgrade repo
git pull

# Pull new images
docker pull rf37535/researchlib

# restart the docker (TODO: only restart if new image exists)
docker rm -f researchlib-jupyter
./start.sh 0 latest
