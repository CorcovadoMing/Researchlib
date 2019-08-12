from ..dataset import *
from ..models import *
from ..runner import *
from ..callbacks import *
from ..utils import *
from ..layers import *
from ..loss import *
from ..metrics import *
from ..search import *
from ..ml import *
from ..pipeline import *
from ..vistool import *
from ..datatool import *
from ..wrapper import *
from pynvml import *
import seaborn as sns
import subprocess

sns.set()
sns.set_style("whitegrid", {'axes.grid': False})

nvmlInit()
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("Using GPU", str(i) + ":", nvmlDeviceGetName(handle).decode('utf-8'))
print("Driver:", nvmlSystemGetDriverVersion().decode('utf-8'))

current_version = '19.08'
print('Researchlib version', current_version)
print('Image version:', os.environ['_RESEARCHLIB_IMAGE_TAG'])
if os.environ['_RESEARCHLIB_IMAGE_TAG'] != current_version:
    print('Researchlib is with different version to the image you are using')
    print(', consider to update the library or the image depend situation.')
else:
    print('Current version is up-to-date!')

# out = subprocess.check_output("git branch -vv", shell=True)
# print(out.decode('utf-8').strip())

# Frontend
def _is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

from ..frontend.dashboard import _Dashboard
import redis
import pickle

if _is_port_in_use(8050):
    print()
    print('* Visit dashboard at http://<ip>:8050')
else:
    r = redis.Redis()
    r.set('progress', 0)
    r.set('desc', '')
    r.set('stage', 'stop')
    r.set('history', pickle.dumps({'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}))
    del r
    dash = _Dashboard()
    dash.start()
    print()
    print('* Dashboard is open at http://<ip>:8050')
