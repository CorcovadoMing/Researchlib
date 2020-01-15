import os

def _set_gpu(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

class config:
    set_gpu = _set_gpu
        