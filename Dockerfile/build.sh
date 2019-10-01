#!/bin/sh

docker build -t rf37535/researchlib:19.09.2 . && docker push rf37535/researchlib:19.09.2 && docker tag rf37535/researchlib:19.09.2 rf37535/researchlib && docker push rf37535/researchlib

