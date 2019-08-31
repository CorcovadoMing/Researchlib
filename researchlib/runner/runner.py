from ..loss import loss_mapping, loss_ensemble
from .history import History
from ..models import GANModel
from .validate import validate_fn
from .preprocessing import PreprocessingDebugger
from .export import _Export
from ..utils import *
from ..utils import _add_methods_from, ParameterManager
from ..benchmark import benchmark
from .save_load import _save_model, _save_optimizer, _load_model, _load_optimizer
from torch.cuda import is_available
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
from apex import amp
import os
import copy
import pandas as pd


from . import init_model
from . import fit
from . import train
from . import validate
from . import gpu_resource_management
from . import predict
from . import set_optimizer

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

    def __init__(self,
                 # Required
                 model,
                 train_loader,
                 # Optional with defualt values
                 test_loader=None,
                 optimizer=None,
                 loss_fn=None,
                 monitor_mode='min',
                 monitor_state='loss',
                 reg_fn={},
                 reg_weights={},
                 **kwargs):

        self.__class__.__runner_settings__ = locals()

        self.experiment_name = ''
        self.checkpoint_path = ''
        self.scheduler = None
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
        self.fp16 = parameter_manager.get_param('fp16', False)
        self.multigpu = parameter_manager.get_param('multigpu', False)
        

        self.default_callbacks = []
        self.preprocessing_list = []
        self.postprocessing_list = []
        self.augmentation_list = []

        
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



    def start_experiment(self, name):
        self.experiment_name = name
        self.checkpoint_path = os.path.join('.', 'checkpoint', self.experiment_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        
    def load_best(self, _id='none'):
        self.load(os.path.join(self.checkpoint_path, 'best_' + _id))

        
    def load_epoch(self, epoch, _id='none'):
        self.load(os.path.join(self.checkpoint_path, 'checkpoint_' + _id + '_epoch_' + str(epoch)))

        
    def load_last(self):
        try:
            self.load_epoch(self.epoch)
        except:
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

            
    def validate(self, metrics=[], callbacks=[]):
        test_loader = self._iteration_pipeline(self.test_loader, 0, inference=True)
        self.preload_gpu()
        try:
            if len(self.default_metrics):
                metrics = self.default_metrics + metrics
            loss_records, matrix_records = self.validate_fn(
                model=self.model,
                test_loader=test_loader,
                loss_fn=self.loss_fn,
                is_cuda=self.is_cuda,
                epoch=self.epoch,
                metrics=metrics,
                callbacks=callbacks,
                inputs=self.inputs)

            for k, v in loss_records.items():
                print(str(k) + ':', str(v))

            if len(metrics) > 0:
                for k, v in matrix_records.records.items():
                    print(str(k) + ':', str(v[-1]))
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

        
    def preprocessing(self, preprocessing_list, debug=False):
        self.preprocessing_list = preprocessing_list
        if debug:
            self.preprocessing_list.append(PreprocessingDebugger())
        return self

    
    def postprocessing(self, postprocessing_list, debug=False):
        self.postprocessing_list = postprocessing_list
        return self

    
    def augmentation(self, augmentation_list, debug=False):
        self.augmentation_list = augmentation_list
        if debug:
            for i in self.augmentation_list:
                i._debug_flag = True
        return self

    # =======================================================================================================
    # following function need to be refined
    def describe(self):
        def _describe_model(model_dict):
            query = {}
            keys = [
                'do_norm', 'pool_freq', 'preact', 'filter_policy', 'filters',
                'type', 'total_blocks', 'op', 'unit'
            ]
            for key, value in model_dict.items():
                if key in keys:
                    query[key] = copy.deepcopy(value)
            target_dict = model_dict['kwargs']
            for key, value in target_dict.items():
                query[key] = copy.deepcopy(value)
            return query

        def _describe_fit(fit_dict):
            query = {}
            keys = ['self_iterative', 'mixup_alpha', 'policy', 'lr', 'epochs']
            for key, value in fit_dict.items():
                if key in keys:
                    query[key] = copy.deepcopy(value)
            return query

        def _get_best_metrics(runner, metrics):
            index = runner.history_.records['saved'].index('*', -1)
            return runner.histroy_.records['val_' + str(metrics)][index]

        keys = [
            'swa', 'swa_start', 'larc', 'fp16', 'augmentation_list',
            'preprocessing_list', 'loss_fn', 'train_loader'
        ]
        query = {}
        for key, value in self.__dict__.items():
            if key in keys:
                query[key] = copy.deepcopy(value)
        try:
            query['loss_fn'] = query['loss_fn'][0].__name__
        except:
            query['loss_fn'] = query['loss_fn'][0].__class__.__name__

        try:
            for i, j in enumerate(query['augmentation_list']):
                query['augmentation_list'][i] = j.__class__.__name__
        except:
            pass

        try:
            for i, j in enumerate(query['preprocessing_list']):
                query['preprocessing_list'][i] = j.__class__.__name__
        except:
            pass

        query['train_loader'] = query['train_loader'].dataset.__class__.__name__

        query['optimizer'] = self.__class__.__runner_settings__['optimizer']
        query['monitor_state'] = self.__class__.__runner_settings__[
            'monitor_state']

        query['num_params'] = self.num_params

        try:
            query['best_state'] = self.__dict__['monitor']
        except:
            query['best_state'] = _get_best_metrics(self,
                                                    query['monitor_state'])

        query['model'] = {}
        for i, j in self.__class__.__model_settings__.items():
            query['model'].setdefault(i, {})
            query['model'][i] = _describe_model(j)
        query['fit'] = {}
        for i, j in self.__class__.__fit_settings__.items():
            query['fit'].setdefault(i, {})
            query['fit'][i] = _describe_fit(j)
        return query

    
    def submit_benchmark(self, category, comments={}, backup=False):
        if type(comments) != dict:
            raise ValueError("Type Error")
        dict_ = self.describe()
        for k, v in comments.items():
            if k in dict_.keys():
                raise ValueError("key is overlapped: {}".forat(k))
            dict_[k] = v
        self.bencher.update_from_runner(
            category, self._date_id, dict_, backup=backup)
