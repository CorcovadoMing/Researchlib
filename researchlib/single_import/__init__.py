from ..dataset import *
from ..models import *
from ..runner import *
from ..callbacks import *
from ..utils import *
from ..layers import *
from ..blocks import *
from ..loss import *
from ..metrics import *
from ..search import *
from ..ml import *
from ..pipeline import *
from ..vistool import *
from ..datatool import *
from ..wrapper import *
from ..benchmark import *
from pynvml import *
import seaborn as sns
import subprocess
from ..frontend.dashboard import _Dashboard
import redis
import pickle

# google api authorization
benchmark()

sns.set()
sns.set_style("whitegrid", {'axes.grid': False})

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

current_version = '19.09'
print('Researchlib version', current_version)
print('Image version:', os.environ['_RESEARCHLIB_IMAGE_TAG'])
if os.environ['_RESEARCHLIB_IMAGE_TAG'] != current_version:
    print('Researchlib is with different version to the image you are using')
    print(', consider to update the library or the image depend situation.')
else:
    print('Current version is up-to-date!')


# Frontend
def _initialize_redis(r, variable, init_value, need_encode=False):
    try:
        result = r.get(variable)
        assert result is not None
        if need_encode:
            result = pickle.loads(result)
    except:
        if need_encode:
            init_value = pickle.dumps(init_value)
        r.set(variable, init_value)


from ..utils import _is_port_in_use
if _is_port_in_use(8050):
    print()
    print('* Visit dashboard at http://<ip>:8050')
else:
    r = redis.Redis()
    _initialize_redis(r, 'progress', 0)
    _initialize_redis(r, 'desc', '')
    _initialize_redis(r, 'stage', 'stop')
    _initialize_redis(
        r,
        'history', {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        },
        need_encode=True)
    _initialize_redis(r, 'experiment', [], need_encode=True)
    del r
    dash = _Dashboard(verbose=False)
    dash.start()
    print()
    print('* Dashboard is open at http://<ip>:8050')
