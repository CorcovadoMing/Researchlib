from pynvml import *
import os


def pre_script(current_version):
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()

    try:
        used_gpus = os.environ["CUDA_VICIBLE_DEVICES"]
    except:
        used_gpus = list(range(deviceCount))
        
    print(f'Available GPUs: (CUDA_VISIBLE_DEVICES={used_gpus})')
    print('==========================================')
    
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print(str(i) + ":", nvmlDeviceGetName(handle).decode('utf-8'))
    print("Driver:", nvmlSystemGetDriverVersion().decode('utf-8'))
    print()
    
    print('Researchlib version', current_version)
    print('Image version:', os.environ['_RESEARCHLIB_IMAGE_TAG'])
    if os.environ['_RESEARCHLIB_IMAGE_TAG'] != current_version:
        print('Researchlib is with different version to the image you are using')
        print(', consider to update the library or the image depend situation.')
    else:
        print('Current version is up-to-date!')