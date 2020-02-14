#!/bin/sh

VERSION=20.02

docker build -t rf37535/researchlib:$VERSION . && docker push rf37535/researchlib:$VERSION && docker tag rf37535/researchlib:$VERSION rf37535/researchlib && docker push rf37535/researchlib:latest

