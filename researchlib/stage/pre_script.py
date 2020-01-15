from pynvml import *
import os


def pre_script(current_version):
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()

    try:
        used_gpus = list(map(int, os.environ['CUDA_VISIBLE_DEVICES']))
    except:
        used_gpus = list(range(deviceCount))
        
    print(f'Selected GPUs:')
    print('==========================================')
    
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        if i in used_gpus:
            token = '[*] '
        else:
            token = '[ ] '
        print(token + str(i) + ":", nvmlDeviceGetName(handle).decode('utf-8'))
    print()
    
    print('Researchlib version', current_version)
    print('Image version:', os.environ['_RESEARCHLIB_IMAGE_TAG'])
    print('Driver:', nvmlSystemGetDriverVersion().decode('utf-8'))
    print()
    
    if os.environ['_RESEARCHLIB_IMAGE_TAG'] != current_version:
        print('Researchlib is with different version to the image you are using')
        print(', consider to update the library or the image depend situation.')
    else:
        print('Current version is up-to-date!')