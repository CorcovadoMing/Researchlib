from ..loss import loss_mapping, loss_ensemble
from .history import History
from .export import _Export
from ..utils import *
from ..utils import _add_methods_from, ParameterManager
from .save_load import _save_checkpoint, _load_checkpoint
from .trainable_params_utils import num_model_params
from torch.cuda import is_available
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import os
import shutil
from ..models import Builder

from . import init_model
from . import fit
from . import train
from . import validate
from . import gpu_resource_management
from . import predict
from . import set_optimizer
from . import describe
from . import calibrate

@_add_methods_from(describe)
@_add_methods_from(set_optimizer)
@_add_methods_from(gpu_resource_management)
@_add_methods_from(init_model)
@_add_methods_from(fit)
@_add_methods_from(train)
@_add_methods_from(validate)
@_add_methods_from(calibrate)
@_add_methods_from(predict)
class Runner:
    __model_settings__ = {}
    __fit_settings__ = {}
    __runner_settings__ = None

    def __init__(
        self,
        model,
        optimizer = None,
        loss_fn = None,
        monitor_mode = 'min',
        monitor_state = 'loss',
        **kwargs
    ):
        self.__class__.__runner_settings__ = locals()
        parameter_manager = ParameterManager(**kwargs)

        self.experiment_name = ''
        self.checkpoint_path = ''
        self.optimizer = None
        self.epoch = 1
        self.is_cuda = is_available()
        
        self._augmentation = None
        self._normalize = None

        self.optimizer_choice = optimizer
        self.export = _Export(self)
        self.history = History()

        self.model = model
        
        # TODO (only for predict)
        self.output_node = parameter_manager.get_param('output_node', required=True)
            
        self.num_params = num_model_params(model)

        self.lookahead = parameter_manager.get_param('lookahead', False)
        self.swa = parameter_manager.get_param('swa', False)
        self.swa_start = parameter_manager.get_param('swa_start', -1)
        self.larc = parameter_manager.get_param('larc', False)
        self.multigpu = parameter_manager.get_param('multigpu', False)

        self.default_callbacks = []

        self.loss_fn = loss_fn

        # Assign monitoring
        self.monitor_mode = monitor_mode
        self.monitor_state = monitor_state
        self.monitor = None

        if monitor_mode == 'min':
            self.monitor = 1e5
            self.monitor_mode = min
        elif monitor_mode == 'max':
            self.monitor = -1e5
            self.monitor_mode = max

        if self.multigpu:
            self.model = DataParallel(self.model)

        # Speedup
        cudnn.benchmark = True

        self.set_optimizer()

        # must verify after all keys get registered
        ParameterManager.verify_kwargs(**kwargs)

    def start_experiment(self, name):
        self.experiment_name = name
        self.checkpoint_path = os.path.join('.', 'checkpoint', self.experiment_name)
        shutil.rmtree(self.checkpoint_path, ignore_errors = True)
        os.makedirs(self.checkpoint_path, exist_ok = True)

    def load_best(self, _id = 'none'):
        if len(self.experiment_name) == 0:
            self.start_experiment('default')
        self.load(os.path.join(self.checkpoint_path, 'best_' + _id))

    def load_epoch(self, epoch, _id = 'none'):
        if len(self.experiment_name) == 0:
            self.start_experiment('default')
        self.load(os.path.join(self.checkpoint_path, 'checkpoint_' + _id + '_epoch_' + str(epoch)))

    def load_last(self):
        try:
            self.load_epoch(self.epoch)
        except:
            # The last epoch is not complete
            self.load_epoch(self.epoch - 1)

    def save(self, path):
        _save_checkpoint(self.model, self.optimizer, path)

    def load(self, path):
        self.model, self.optimizer = _load_checkpoint(
            self.model, self.optimizer, self.multigpu, path
        )