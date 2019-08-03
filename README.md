# researchlib

### Setup the environment
1. pull the docker container for researchlib
```
    docker pull rf37535/researchlib
```
2. In the researchlib, start the docker environemnt using the `start.sh` script, you may need to grand the execution permission to `start.sh` first. The `start.sh` works like `./start.sh <gpu_id> <container version`
```
    cd researchlib
    ./start 0 latest
```
3. Open the browser to the port `8899` and the password `0000` to dive into the Jupyter environment. That it!
