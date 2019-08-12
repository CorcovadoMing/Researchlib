#!/bin/sh

redis-server /etc/redis/redis.conf &
xvfb-run -s '-screen 0 1400x900x24' jupyter lab --ip 0.0.0.0 --port 8899 --allow-root --no-browser
