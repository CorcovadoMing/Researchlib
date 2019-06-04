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


from pynvml import *
nvmlInit()

deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("Using GPU", str(i)+":", nvmlDeviceGetName(handle).decode('utf-8'))

print("Driver:", nvmlSystemGetDriverVersion().decode('utf-8'))

# deviceCount = nvmlDeviceGetCount()
# for i in range(deviceCount):
#     handle = nvmlDeviceGetHandleByIndex(i)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print('Total:', info.total)
#     print('Used:', info.used)
#     s = nvmlDeviceGetUtilizationRates(handle)
#     print(s.gpu, s.memory)