#!/bin/sh

docker build -t rf37535/researchlib:19.09 . && docker push rf37535/researchlib:19.09 && docker tag rf37535/researchlib:19.09 rf37535/researchlib && docker push rf37535/researchlib

