# Researchlib

This was an internal experimental library aim to rapidly deal with different type of domain applications. This library will not get further updates anymore. It's has been replaced by [CaTorch](https://github.com/CorcovadoMing/catorch) which is currently internal only. Will be public soon.


### Setup the environment
1. Pull the docker container for researchlib
```
    docker pull rf37535/researchlib
```
2. Clone the code to the place you prefer
```
    git clone rf37535/researchlib
```
3. In the researchlib, start the docker environemnt using the `start.sh` script, you may need to grand the execution permission to `start.sh` first. The `start.sh` works like `./start.sh <gpu_id> <container version`
```
    cd researchlib
    ./start 0 latest
```
4. Open the browser to the port `8899` and the password `0000` to dive into the Jupyter environment. That's it!
