from ...dataset import *
from ...models import *
from ...runner import *
from ...callback import *
from ...utils import *
from ...ops import *
from ...blocks import *
from ...loss import *
from ...metrics import *
from ...search import *
from ...mlkit import *
from ...pipeline import *
from ...viskit import *
from ...datakit import *
from ...wrapper import *
from ...reproducer import *
from ...experiments import *
from ...preset import *

import seaborn as sns

from ..logo import logo
from ..pre_script import pre_script
from ..version import version
from ..initial_frontend import initial_frontend

# Print logo
logo()

# Google api authorization
uploader()

# Pre-script (GPU info, etc.,)
pre_script(version)
sns.set()
sns.set_style("whitegrid", {'axes.grid': False})


# Frontend
initial_frontend()