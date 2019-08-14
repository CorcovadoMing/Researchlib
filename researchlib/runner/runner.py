from .history import *
from ..utils import *
from ..metrics import *
from ..loss import *
from ..layers import *
from torch.optim import *
from ..models import *
from ..callbacks import *

# -------------------------------------------------------
from .adafactor import Adafactor
from .validate import validate_fn
from .preprocessing import PreprocessingDebugger
from .export import _Export
from ..utils import _add_methods_from, _get_iteration
from .save_load import _save_model, _save_optimizer, _load_model, _load_optimizer
from torch.cuda import is_available
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
# from apex import amp
import torchcontrib
import os
import datetime
import pandas as pd
from adabound import AdaBound
from .larc import LARC

from . import init_model
from . import fit
from . import train
from . import validate
from . import gpu_resource_management
from . import predict


@_add_methods_from(gpu_resource_management)
@_add_methods_from(init_model)
@_add_methods_from(fit)
@_add_methods_from(train)
@_add_methods_from(validate)
@_add_methods_from(predict)
class Runner:
    def __init__(self,
                 model=None,
                 train_loader=None,
                 test_loader=None,
                 optimizer=None,
                 loss_fn=None,
                 reg_fn={},
                 reg_weights={},
                 monitor_mode='min',
                 monitor_state='loss',
                 fp16=False,
                 multigpu=False,
                 larc=False,
                 ema=-1,
                 ema_start=100,
                 swa=False,
                 swa_start=20):

        self.experiment_name = ''
        self.checkpoint_path = ''
        self.scheduler = None
        self.ema = ema
        self.ema_start = ema_start
        self.swa = swa
        self.swa_start = swa_start
        self.larc = larc
        self.export = _Export(self)
        self.epoch = 1
        self.is_cuda = is_available()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.inputs = 1
        if type(train_loader) == tuple:
            self.train_loader = train_loader[0]
            self.inputs = train_loader[1]
        if type(test_loader) == tuple:
            self.test_loader = test_loader[0]
            self.inputs = test_loader[1]
        self.history_ = History()
        self.fp16 = fp16

        self.default_callbacks = []

        self.preprocessing_list = []
        self.postprocessing_list = []
        self.augmentation_list = []

        # Assign loss function
        #
        # self.loss_fn
        # self.default_metrics
        self.loss_fn = []
        self.default_metrics = []

        # --------------------------------------------------------------------------------------------------------------------------------
        def _process_loss_fn(loss_fn):
            if type(loss_fn) == type({}):
                process_func = loss_ensemble
            else:
                process_func = loss_mapping
            return process_func(loss_fn)

        if type(loss_fn) == type([]):
            for lf in loss_fn:
                _loss_fn, _default_metrics = _process_loss_fn(lf)
                self.loss_fn.append(_loss_fn)
                self.default_metrics += _default_metrics
        else:
            _loss_fn, _default_metrics = _process_loss_fn(loss_fn)
            self.loss_fn.append(_loss_fn)
            self.default_metrics += _default_metrics
        # --------------------------------------------------------------------------------------------------------------------------------

        # Assign monitoring
        #
        # self.monitor_mode
        # self.monitor_state
        # self.monitor
        self.monitor_mode = monitor_mode
        self.monitor_state = monitor_state
        self.monitor = None
        # --------------------------------------------------------------------------------------------------------------------------------
        if monitor_mode == 'min':
            self.monitor = 1e9
            self.monitor_mode = min
        elif monitor_mode == 'max':
            self.monitor = -1e9
            self.monitor_mode = max
        # --------------------------------------------------------------------------------------------------------------------------------

        self.reg_fn = reg_fn
        for key in self.reg_fn:
            if type(reg_fn[key]) == str:
                fn, _, _, _, _ = loss_mapping(reg_fn[key])
                reg_fn[key] = fn

        self.reg_weights = reg_weights

        # Model
        self.model = model
        self.multigpu = multigpu
        if type(model) == GANModel:
            self.loss_fn[0].set_model(self.model)
            self.default_metrics = [InceptionScore(), FID()]

        if self.multigpu: self.model = DataParallel(self.model)

        if optimizer is not None: self.set_optimizer(optimizer)

        cudnn.benchmark = True

    # ===================================================================================================
    # ===================================================================================================
    def set_optimizer(self, optimizer):
        def _assign_optim(model, optimizer, larc, swa):
            # if there are learnable loss parameters
            loss_params = []
            for i in self.loss_fn:
                try:
                    loss_params += i.parameters()
                except:
                    pass

            if optimizer == 'adam':
                optimizer = Adam(list(model.parameters()) + loss_params,
                                 betas=(0.9, 0.999))
            elif optimizer == 'adam_gan':
                optimizer = Adam(list(model.parameters()) + loss_params,
                                 betas=(0., 0.999))
            elif optimizer == 'sgd':
                optimizer = SGD(list(model.parameters()) + loss_params,
                                lr=1e-1,
                                momentum=0.9)
            elif optimizer == 'nesterov':
                optimizer = SGD(list(model.parameters()) + loss_params,
                                lr=1e-2,
                                momentum=0.9,
                                nesterov=True)
            elif optimizer == 'rmsprop':
                optimizer = RMSprop(list(model.parameters()) + loss_params)
            elif optimizer == 'adabound':
                optimizer = AdaBound(list(model.parameters()) + loss_params,
                                     lr=1e-3,
                                     final_lr=0.1)
            elif optimizer == 'adagrad':
                optimizer = Adagrad(list(model.parameters()) + loss_params)
            elif optimizer == 'adafactor':
                optimizer = Adafactor(list(model.parameters()) + loss_params,
                                      lr=1e-3)
            if larc:
                optimizer = LARC(optimizer)
            if swa:
                optimizer = torchcontrib.optim.SWA(optimizer)
            return optimizer

        if type(self.model) == GANModel:
            if type(optimizer) == list or type(optimizer) == tuple:
                self.optimizer = [
                    _assign_optim(self.model.discriminator, optimizer[1],
                                  self.larc, self.swa),
                    _assign_optim(self.model.generator, optimizer[0],
                                  self.larc, self.swa)
                ]
            else:
                self.optimizer = [
                    _assign_optim(self.model.discriminator, optimizer,
                                  self.larc, self.swa),
                    _assign_optim(self.model.generator, optimizer, self.larc,
                                  self.swa)
                ]
        else:
            self.optimizer = _assign_optim(self.model, optimizer, self.larc,
                                           self.swa)


#         self.model, self.optimizer = amp.initialize(self.model,
#                                                     self.optimizer,
#                                                     opt_level="O2",
#                                                     enabled=self.fp16)

    def start_experiment(self, name):
        self.experiment_name = name
        self.experiment_name += str(datetime.datetime.utcnow()).replace(
            ' ', '_')
        self.checkpoint_path = os.path.join('.', 'checkpoint',
                                            self.experiment_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def load_best(self, _id='none'):
        self.load(os.path.join(self.checkpoint_path, 'best_' + _id))

    def load_epoch(self, epoch, _id='none'):
        self.load(
            os.path.join(self.checkpoint_path,
                         'checkpoint_' + _id + '_epoch_' + str(epoch)))

    def load_last(self):
        try:
            self.load_epoch(self.epoch)
        except:
            self.load_epoch(self.epoch - 1)

    def resume_best(self):
        self.resume(os.path.join(self.checkpoint_path, 'best.h5'))

    def resume_epoch(self, epoch, _id='none'):
        self.resume(
            os.path.join(self.checkpoint_path,
                         'checkpoint_' + _id + '_epoch_' + str(epoch)))

    def resume_last(self):
        try:
            self.resume_epoch(self.epoch)
        except:
            self.resume_epoch(self.epoch - 1)

    def report(self):
        print('Experiment:', self.experiment_name)
        print('Checkpoints are saved in', self.checkpoint_path)
        df = pd.DataFrame.from_dict(self.history_.records)
        df.index += 1
        df.columns.name = 'Epoch'
        return df

    def history(self, plot=True):
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(24, 8))
            legends = [[], []]
            for key in self.history_.records:
                if 'loss' in key:
                    legends[0].append(key)
                    ax[0].plot(self.history_.records[key])
                else:
                    legends[1].append(key)
                    ax[1].plot(self.history_.records[key])
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epochs")
            ax[0].set_ylabel("Loss")
            ax[0].legend(legends[0])
            ax[1].set_title("Accuracy")
            ax[1].set_xlabel("Epochs")
            ax[1].set_ylabel("Accuracy")
            ax[1].legend(legends[1])
            plt.show()
        else:
            return self.history_

    def validate(self, metrics=[], callbacks=[]):
        self.preload_gpu()
        try:
            if len(self.default_metrics):
                metrics = self.default_metrics + metrics
            loss_records, matrix_records = self.validate_fn(
                model=self.model,
                test_loader=self.test_loader,
                loss_fn=self.loss_fn,
                is_cuda=self.is_cuda,
                epoch=1,
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
            self.unload_gpu(unload_data=False)

    def save(self, path):
        # TODO: more efficient to save optimizer (save only the last/best?)
        _save_model(self.model, path)
        #_save_optimizer(self.optimizer, path)

    def load(self, path):
        self.model = _load_model(self.model, path, self.multigpu)

    def resume(self, path):
        self.load(path)
        _load_optimizer(self.optimizer, path)

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

    def find_lr(self, mixup_alpha=0, plot=False, callbacks=[]):
        _save_model(self.model, 'find_lr_tmp.h5')
        try:
            loss, _ = self.trainer(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                reg_fn=self.reg_fn,
                reg_weights=self.reg_weights,
                epoch=1,
                augmentor=None,
                is_cuda=self.is_cuda,
                mixup_alpha=mixup_alpha,
                callbacks=[
                    LRRangeTest(_get_iteration(self.train_loader),
                                cutoff_ratio=10)
                ] + callbacks,
                metrics=[],
                inputs=self.inputs)

            step = (10 / 1e-9)**(1 / _get_iteration(self.train_loader))
            self.loss_history = []
            self.lr_history = []
            for i, j in enumerate(loss):
                self.loss_history.append(j)
                self.lr_history.append(1e-9 * (step**i))
            if plot:
                plot_utils(self.loss_history, self.lr_history)
        except Exception as e:
            print('Error:', e)
        finally:
            self.model = _load_model(self.model, 'find_lr_tmp.h5',
                                     self.multigpu)
