from .train import train_fn
from .test import test_fn
from .history import *
from ..utils import *
from ..metrics import *
from ..loss import *
from ..layers import *
from .save_load import _save_model, _load_model
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import *
import torch.backends.cudnn as cudnn
from apex import amp
from ..models import *

from ..callbacks import *
from ..utils import _add_methods_from, _get_iteration

# -------------------------------------------------------
from . import init_model
from . import fit

@_add_methods_from(init_model, fit)
class Runner:
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, loss_fn=None, reg_fn={}, reg_weights={}, monitor_mode='min', monitor_state='loss', fp16=False, multigpu=False):
        '''
            Multi-model supported
        '''
        self.is_cuda = torch.cuda.is_available()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.history_ = History()
        self.multi_model = False
        self.cam_model = None
        
        self.default_callbacks = CyclicalLR(_get_iteration(self.train_loader))
        
        self.trainer = train_fn
        self.tester = test_fn
        
        self.default_metrics = None
        
        # Assign loss function
        # 
        # self.loss_fn
        # self.require_long_
        # self.keep_x_shape_
        # self.keep_y_shape_
        # self.default_metrics
        self.loss_fn = []
        self.require_long_ = []
        self.keep_x_shape_ = []
        self.keep_y_shape_ = []
        self.default_metrics = None
        # --------------------------------------------------------------------------------------------------------------------------------
        def _process_loss_fn(loss_fn):
            if type(loss_fn) == type({}):
                process_func = loss_ensemble
            else:
                process_func = loss_mapping
            return process_func(loss_fn)
        
        if type(loss_fn) == type([]):
            for lf in loss_fn:
                _loss_fn, require_long_, keep_x_shape_, keep_y_shape_, self.default_metrics = _process_loss_fn(lf)
                self.loss_fn.append(_loss_fn)
                self.require_long_ += require_long_
                self.keep_x_shape_ += keep_x_shape_
                self.keep_y_shape_ += keep_y_shape_
        else:
            _loss_fn, require_long_, keep_x_shape_, keep_y_shape_, self.default_metrics = _process_loss_fn(loss_fn)
            self.loss_fn.append(_loss_fn)
            self.require_long_ += require_long_
            self.keep_x_shape_ += keep_x_shape_
            self.keep_y_shape_ += keep_y_shape_
        # --------------------------------------------------------------------------------------------------------------------------------
        
        
        
        # Assign optimizer
        # 
        # self.optimizer
        # --------------------------------------------------------------------------------------------------------------------------------
        def _assign_optim(model, optimizer):
            if optimizer == 'adam':
                return Adam(model.parameters(), betas=(0.9, 0.99))
            elif optimizer == 'sgd':
                return SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-5)
            elif optimizer == 'rmsprop':
                return RMSprop(model.parameters(), weight_decay=5e-5)
            else:
                return optimizer
        
        if type(model) == GANModel:
            self.optimizer = [_assign_optim(model.discriminator, optimizer), _assign_optim(model.generator, optimizer)]
        else:
            self.optimizer = _assign_optim(model, optimizer)
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
        if self.default_metrics:
            self.monitor_state = 'metrics'
            self.monitor_mode = 'max'
        # --------------------------------------------------------------------------------------------------------------------------------
        if monitor_mode == 'min':
            self.monitor = 1e9
            self.monitor_mode = min
        elif monitor_mode == 'max':
            self.monitor = 0
            self.monitor_mode = max
        # --------------------------------------------------------------------------------------------------------------------------------
        
        self.reg_fn = reg_fn
        for key in self.reg_fn:
            if type(reg_fn[key]) == type(''):
                fn, _, _, _, _ = loss_mapping(reg_fn[key])
                reg_fn[key] = fn
        
        self.reg_weights = reg_weights
        
        cudnn.benchmark = True
        
        # Model
        self.model = model
        if type(model) == GANModel:
            self.loss_fn[0].set_model(self.model)
        
        # FP16
        if self.is_cuda:
            self.model = self.model.cuda()
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2", enabled=fp16)
        
        # Multi GPU
        self.multigpu = multigpu
        if self.multigpu:
            self.model = nn.DataParallel(self.model)
        
    
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
        '''
            Multi-model supported
        '''
        if self.default_metrics:
            metrics = [self.default_metrics] + metrics
        
        loss_records, matrix_records = self.tester(model=self.model, 
                                                            test_loader=self.test_loader, 
                                                            loss_fn=self.loss_fn, 
                                                            is_cuda=self.is_cuda,
                                                            epoch=1,
                                                            require_long=self.require_long_, 
                                                            keep_x_shape=self.keep_x_shape_,
                                                            keep_y_shape=self.keep_y_shape_,
                                                            metrics=metrics,
                                                            callbacks=callbacks)
        if len(metrics) > 0: 
            print(loss_records, list(matrix_records.records.values())[-1][-1])
        else:
            print(loss_records)
    
    

    def save(self, path):
        _save_model(self.model, path)
    
    def load(self, path):
        self.model = _load_model(self.model, path, self.multigpu)

    def find_lr(self, mixup_alpha=0, plot=False, callbacks=[]):
        _save_model(self.model, 'tmp.h5')
        try:
            loss, _ = self.trainer(model=self.model, 
                                train_loader=self.train_loader, 
                                optimizer=self.optimizer, 
                                loss_fn=self.loss_fn,
                                reg_fn=self.reg_fn,
                                reg_weights=self.reg_weights,
                                epoch=1,
                                augmentor=None,
                                is_cuda=self.is_cuda, 
                                require_long=self.require_long_, 
                                keep_x_shape=self.keep_x_shape_,
                                keep_y_shape=self.keep_y_shape_,
                                mixup_alpha=mixup_alpha,
                                callbacks=[LRRangeTest(_get_iteration(self.train_loader), cutoff_ratio=10)]+callbacks,
                                metrics=[])
            
            step = (10 / 1e-9) ** (1 / _get_iteration(self.train_loader))
            self.loss_history = []
            self.lr_history = []
            for i, j in enumerate(loss):    
                self.loss_history.append(j)
                self.lr_history.append(1e-9 * (step ** i))
            if plot:
                plot_utils(self.loss_history, self.lr_history)
        except Exception as e: print('Error:', e)
        finally: self.model = _load_model(self.model, 'tmp.h5', self.multigpu)
    
    def cam(self, vx, final_layer, out_filters, classes):
        if not self.cam_model:
            self.cam_model = nn.Sequential(
                    *list(self.model.children())[:final_layer+1], 
                    nn.Conv2d(out_filters, classes, 1), 
                    AdaptiveConcatPool2d(1),
                    nn.Conv2d(classes*2, classes, 1),
                    Flatten(),
                    nn.LogSoftmax(1)
                ).cuda()
            self.cam_feature_layer = -5
            module_trainable(self.cam_model[:self.cam_feature_layer], False)
            #self.fit_onecycle()
            r = Runner(self.cam_model, self.train_loader, self.test_loader, 'adam', 'nll', fp16=False)
            r.fit(10, 1e-3)
            
        self.cam_feature = SaveFeatures(self.cam_model[self.cam_feature_layer])
        py = self.cam_model(vx.cuda().float())
        py = F.softmax(py, dim=-1)
        py = py.detach().cpu().numpy()[0]
        feat = self.cam_feature.features[0].detach().cpu().numpy()
        feat = np.maximum(0, feat)
        f2 = np.dot(np.rollaxis(feat,0,3), py)
        f2 -= f2.min()
        f2 /= f2.max()
        dx = vx.cpu().numpy().transpose(0,2,3,1)[0]
        #import skimage
        plt.axis('off')
        #plt.imshow(dx)
        #ss = skimage.transform.resize(f2, dx.shape[:2])
        plt.imshow(f2, alpha=0.5, cmap='hot')
        module_trainable(self.model, True)
        self.cam_feature.remove()
