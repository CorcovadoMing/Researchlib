from ..loss import loss_mapping, loss_ensemble
from .history import History
from ..models import GANModel
from .export import _Export
from .prefetch import *
from ..utils import *
from ..utils import _add_methods_from, ParameterManager
from ..benchmark import benchmark
from .save_load import _save_model, _save_optimizer, _load_model, _load_optimizer
from torch.cuda import is_available
from torch.nn import DataParallel
from torchvision import transforms
import torch.backends.cudnn as cudnn
from apex import amp
import os
import pandas as pd
import pickle

from . import init_model
from . import fit
from . import train
from . import validate
from . import gpu_resource_management
from . import predict
from . import set_optimizer
from . import describe

@_add_methods_from(describe)
@_add_methods_from(set_optimizer)
@_add_methods_from(gpu_resource_management)
@_add_methods_from(init_model)
@_add_methods_from(fit)
@_add_methods_from(train)
@_add_methods_from(validate)
@_add_methods_from(predict)
class Runner:
    __model_settings__ = {}
    __fit_settings__ = {}
    __runner_settings__ = None

    def __init__(
        self,
        # Required
        model,
        train_loader,
        # Optional with defualt values
        test_loader = None,
        optimizer = None,
        loss_fn = None,
        monitor_mode = 'min',
        monitor_state = 'loss',
        reg_fn = {},
        reg_weights = {},
        **kwargs
    ):

        self.__class__.__runner_settings__ = locals()

        self.experiment_name = ''
        self.checkpoint_path = ''
        self.optimizer = None
        self.epoch = 1
        self.is_cuda = is_available()

        self.optimizer_choice = optimizer
        self.export = _Export(self)
        self.history_ = History()
        self.bencher = benchmark()
        self._date_id = self.bencher.get_date()

        self.model = model
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inputs = 1
        if type(train_loader) == tuple:
            self.train_loader = train_loader[0]
            self.inputs = train_loader[1]
        if type(test_loader) == tuple:
            self.test_loader = test_loader[0]
            self.inputs = test_loader[1]

        parameter_manager = ParameterManager(**kwargs)

        self.lookahead = parameter_manager.get_param('lookahead', False)
        self.swa = parameter_manager.get_param('swa', False)
        self.swa_start = parameter_manager.get_param('swa_start', -1)
        self.larc = parameter_manager.get_param('larc', False)
        self.multigpu = parameter_manager.get_param('multigpu', False)

        self.default_callbacks = []

        # Assign loss function
        self.loss_fn = []
        self.default_metrics = []

        def _process_loss_fn(loss_fn):
            process_func = loss_ensemble if type(loss_fn) == dict else loss_mapping
            return process_func(loss_fn)

        if type(loss_fn) == list:
            for lf in loss_fn:
                _loss_fn, _default_metrics = _process_loss_fn(lf)
                self.loss_fn.append(_loss_fn)
                self.default_metrics += _default_metrics
        else:
            _loss_fn, _default_metrics = _process_loss_fn(loss_fn)
            self.loss_fn.append(_loss_fn)
            self.default_metrics += _default_metrics

        # Assign monitoring
        self.monitor_mode = monitor_mode
        self.monitor_state = monitor_state
        self.monitor = None

        if monitor_mode == 'min':
            self.monitor = 1e9
            self.monitor_mode = min
        elif monitor_mode == 'max':
            self.monitor = -1e9
            self.monitor_mode = max

        # Regulariation (Need to be check)
        self.reg_fn = reg_fn
        self.reg_weights = reg_weights
        for key in self.reg_fn:
            if type(reg_fn[key]) == str:
                fn, _, = loss_mapping(reg_fn[key])
                reg_fn[key] = fn

        # Model
        if type(model) == GANModel:
            self.loss_fn[0].set_model(self.model)
            self.default_metrics = [InceptionScore(), FID()]

        if self.multigpu:
            self.model = DataParallel(self.model)

        # Speedup
        cudnn.benchmark = True

        # must verify after all keys get registered
        ParameterManager.verify_kwargs(**kwargs)

    def start_experiment(self, name):
        self.experiment_name = name
        self.checkpoint_path = os.path.join('.', 'checkpoint', self.experiment_name)
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

    def report(self):
        print('Experiment:', self.experiment_name)
        print('Checkpoints are saved in', self.checkpoint_path)
        df = pd.DataFrame.from_dict(self.history_.records)
        df.index += 1
        df.columns.name = 'Epoch'
        return df

    def eval(self):
        self.model.eval()
        _switch_swa_mode(self.optimzier)

    def train(self):
        self.model.train()
        _switch_swa_mode(self.optimzier)

    def validate(self, metrics = [], callbacks = [], prefetch=True, **kwargs):
        buffered_epochs = 2
        test_loader = self.test_loader.get_generator(epochs=buffered_epochs)
        self.test_loader_length = len(test_loader)
        test_loader = self._iteration_pipeline(test_loader)
        test_loader = BackgroundGenerator(test_loader)
        self.preload_gpu()
        try:
            if len(self.default_metrics):
                metrics = self.default_metrics + metrics

            loss_record = self.validate_fn(test_loader, metrics)

            print(loss_record)
            
            if len(metrics) > 0:
                for m in metrics:
                    for k, v in m.output().items():
                        print(str(k) + ':', str(v))
        except:
            raise
        finally:
            self.unload_gpu()

    def save(self, path):
        _save_model(self.model, path)
        # TODO: more efficient to save optimizer (save only the last/best?)
        #_save_optimizer(self.optimizer, path)

    def load(self, path):
        self.model = _load_model(self.model, path, self.multigpu)
        
    def normalize(self, local=False):
        self.train_loader._set_normalizer(local)
        if self.test_loader:
            self.test_loader._set_normalizer(local)
        return self

    def augmentation(self, augmentation_list, include_y=False):
        self.train_loader._set_augmentor(augmentation_list, include_y)
        return self

    def submit_benchmark(self, category, comments = {}, backup = False):
        if type(comments) != dict:
            raise ValueError("Type Error")
        dict_ = self.describe()
        for k, v in comments.items():
            if k in dict_.keys():
                raise ValueError("key is overlapped: {}".forat(k))
            dict_[k] = v
        self.bencher.update_from_runner(category, self._date_id, dict_, backup = backup)
