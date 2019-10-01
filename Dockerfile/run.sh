#!/bin/sh

redis-server /etc/redis/redis.conf &
xvfb-run -s '-screen 0 1400x900x24' jupyter lab --ip 0.0.0.0 --port 8899 --NotebookApp.password='sha1:9483100109c0:ec3eff89acf3b78b73cc0dda06c31e7ebd5bf17d' --allow-root --no-browser
