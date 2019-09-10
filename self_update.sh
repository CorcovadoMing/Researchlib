#!/bin/sh

# Upgrade repo
git pull

# Pull new images
docker pull rf37535/researchlib:$2

# restart the docker (TODO: only restart if new image exists)
docker rm -f researchlib-jupyter
./start.sh $1 $2
