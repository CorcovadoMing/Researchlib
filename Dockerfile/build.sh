#!/bin/sh

docker build -t rf37535/researchlib:19.11.0 . && docker push rf37535/researchlib:19.11.0 && docker tag rf37535/researchlib:19.11.0 rf37535/researchlib && docker push rf37535/researchlib:latest

