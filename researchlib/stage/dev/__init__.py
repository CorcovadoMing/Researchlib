__settings = {}

import warnings
warnings.filterwarnings('ignore')

from ...dataset import *
from ...models import *
from ...runner import *
from ...callback import *
from ...utils import *
from ...ops import *
from ...blocks import *
from ...loss import *
from ...metrics import *
from ...mlkit import *
from ...pipeline import *
from ...viskit import *
from ...datakit import *
from ...reproducer import *
from ...experiments import *
from ...preset import *
from ...regularizer import *

import seaborn as sns

from ..logo import logo
from ..pre_script import pre_script
from ..version import version, checkversion
from ..initial_frontend import initial_frontend

# Print logo
logo()

# Google api authorization
uploader()

# Pre-script (GPU info, etc.,)
pre_script(version, checkversion)
sns.set()
sns.set_style("whitegrid", {'axes.grid': False})

# Frontend
initial_frontend()

warnings.filterwarnings('once')