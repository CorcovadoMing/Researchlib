from .train import *
from .test import *
from .history import *
from ..io import *
from ..callbacks import *
from ..utils import *
from ..metrics import *
from ..loss import *
from ..layers import *
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import *
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
        
class Runner:
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, loss_fn=None, monitor_mode='min', monitor_state='metrics'):
        '''
            Multi-model supported
        '''
        self.is_cuda = torch.cuda.is_available()
        self.require_long_ = False
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.keep_x_shape_ = False
        self.keep_y_shape_ = False
        self.require_data_ = False
        self.history_ = History()
        self.multi_model = False
        self.cam_model = None
        
        # Monitor best model
        self.monitor_mode = monitor_mode
        self.monitor_state = monitor_state
        self.monitor = None
        
        self.default_callbacks = CyclicalLR(len(train_loader))
        
        self.trainer = train
        self.tester = test
        
        self.default_metrics = None
        
        if type(model) == type([]):
            self.multi_model = True
        
        # Check if cuda available (could be overwrite)
        if self.is_cuda:
            if self.multi_model:
                self.model = [i.cuda for i in model]
            else:
                self.model = model.cuda()
        else:
            self.model = model
            
        # Assign loss function
        if type(loss_fn) == type({}):
            self.loss_fn, self.require_long_, self.keep_x_shape_, self.keep_y_shape_, self.require_data_, self.default_metrics = loss_ensemble(loss_fn)
        else:
            self.loss_fn, self.require_long_, self.keep_x_shape_, self.keep_y_shape_, self.require_data_, self.default_metrics = loss_mapping(loss_fn)
            
        # Assign optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(model.parameters(), betas=(0.9, 0.99), amsgrad=True)
        elif optimizer == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(model.parameters())
        else:
            self.optimizer = optimizer
        
        if monitor_mode == 'min':
            self.monitor = 1e9
            self.monitor_mode = min
        elif monitor_mode == 'max':
            self.monitor = 0
            self.monitor_mode = max
            
        cudnn.benchmark = True
        
    def summary(self):
        input_shape = self.train_loader.dataset.train_data.shape
        if len(input_shape) > 4:
            input_shape = (input_shape[-1], input_shape[-3], input_shape[-2])
        else:
            input_shape = (1, input_shape[-2], input_shape[-1])
        try:
            summary(self.model, input_shape)
        except:
            pass
        
    def set_cyclical_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = CyclicalLR(len(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            return [self.default_callbacks]
        else:
            return []
            
    def set_sgdr_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = SGDR(len(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            self.default_callbacks.length = 1
            return [self.default_callbacks]
        else:
            return []
    
    def set_onecycle_(self, lr):
        if self.default_callbacks:
            self.default_callbacks = OneCycle(len(self.train_loader))
            self.default_callbacks.max_lr = lr
            self.default_callbacks.acc_iter = 0
            return [self.default_callbacks]
        else:
            return []
    
    def fit_onecycle(self, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        callbacks = self.set_onecycle_(lr) + callbacks
        self.fit_(1, lr, augmentor, mixup_alpha, metrics, callbacks)
        
    def fit_cycle(self, cycles, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        total_epochs = int(cycles*(1+cycles)/2)
        callbacks = self.set_sgdr_(lr) + callbacks
        self.fit_(total_epochs, lr, augmentor, mixup_alpha, metrics, callbacks)
        
    def fit(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        callbacks = self.set_cyclical_(lr) + callbacks
        self.fit_(epochs, lr, augmentor, mixup_alpha, metrics, callbacks)

    def fit_(self, epochs, lr=1e-3, augmentor=None, mixup_alpha=0, metrics=[], callbacks=[]):
        if self.default_metrics:
            metrics = [self.default_metrics] + metrics
        
        for epoch in tqdm(range(1, epochs + 1)):
            epoch_str = str(epoch)
        
            for callback_func in callbacks:
                callback_func.on_epoch_begin(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=epoch)
            
            loss_records, matrix_records = self.trainer(model=self.model, 
                                                        train_loader=self.train_loader, 
                                                        optimizer=self.optimizer, 
                                                        loss_fn=self.loss_fn, 
                                                        epoch=epoch, 
                                                        augmentor=augmentor,
                                                        is_cuda=self.is_cuda, 
                                                        require_long=self.require_long_, 
                                                        keep_x_shape=self.keep_x_shape_,
                                                        keep_y_shape=self.keep_y_shape_,
                                                        require_data=self.require_data_,
                                                        mixup_alpha=mixup_alpha,
                                                        callbacks=callbacks,
                                                        metrics=metrics)
            
            self.history_.add({'train_loss': sum(loss_records)/len(loss_records)})
            self.history_ += matrix_records
            
            
            for callback_func in callbacks:
                callback_func.on_epoch_end(model=self.model, 
                                            train_loader=self.train_loader, 
                                            optimizer=self.optimizer,
                                            epoch=epoch)
            
            if self.test_loader:
                loss_records, matrix_records = self.tester(model=self.model, 
                                                            test_loader=self.test_loader, 
                                                            loss_fn=self.loss_fn, 
                                                            is_cuda=self.is_cuda,
                                                            epoch=epoch,
                                                            require_long=self.require_long_, 
                                                            require_data=self.require_data_, 
                                                            keep_x_shape=self.keep_x_shape_,
                                                            keep_y_shape=self.keep_y_shape_,
                                                            metrics=metrics,
                                                            callbacks=callbacks)
                self.history_.add({'val_loss': loss_records})
                self.history_ += matrix_records
                
                cri = None
                if self.monitor_state == 'metrics':
                    cri = list(matrix_records.records.values())[-1][-1]
                else:
                    cri = loss_records
                
                # Checkpoint
                if self.monitor_mode(cri, self.monitor) == cri:
                    self.monitor = cri
                    save_model(self.model, 'checkpoint.h5')
                    epoch_str += '*'
                
            state = []
            fs = '{:^14}'
            if epoch == 1:
                if self.test_loader:
                    print(('{:^10}'+(fs*len(self.history_.records.keys()))).format('Epochs', *self.history_.records.keys()))
                    print('================================================================')
                else:
                    print(('{:^10}'+(fs*len(self.history_.records.keys()))).format('Epochs', *self.history_.records.keys()))
                    print('==============================') # Untested
            state.append('{:^10}'.format(epoch_str))
            for i in self.history_.records:
                state.append('{:^14.4f}'.format(self.history_.records[i][-1]))
            print(''.join(state))
            
    
    
    def history(self, plot=False):
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
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
                                                            require_data=self.require_data_, 
                                                            keep_x_shape=self.keep_x_shape_,
                                                            keep_y_shape=self.keep_y_shape_,
                                                            metrics=metrics,
                                                            callbacks=callbacks)
        if len(metrics) > 0: 
            print(loss_records, list(matrix_records.records.values())[-1][-1])
        else:
            print(loss_records)

    def save(self, path):
        '''
            Save model to path
            Multi-model supported
        '''
        save_model(self.model, path)
    
    def load(self, path):
        '''
            Load model from path
            Multi-model supported
        '''
        load_model(self.model, path)

    def find_lr(self, mixup_alpha=0, plot=False, callbacks=[]):
        '''
            Multi-model supported
        '''
        save_model(self.model, 'tmp.h5')
        try:
            loss, _ = self.trainer(model=self.model, 
                                train_loader=self.train_loader, 
                                optimizer=self.optimizer, 
                                loss_fn=self.loss_fn, 
                                epoch=1,
                                augmentor=None,
                                is_cuda=self.is_cuda, 
                                require_long=self.require_long_, 
                                keep_x_shape=self.keep_x_shape_,
                                keep_y_shape=self.keep_y_shape_,
                                require_data=self.require_data_,
                                mixup_alpha=mixup_alpha,
                                callbacks=[LRRangeTest(len(self.train_loader), cutoff_ratio=10)]+callbacks,
                                metrics=[])
            
            step = (10 / 1e-9) ** (1 / len(self.train_loader))
            self.loss_history = []
            self.lr_history = []
            for i, j in enumerate(loss):    
                self.loss_history.append(j)
                self.lr_history.append(1e-9 * (step ** i))
            if plot:
                plot_utils(self.loss_history, self.lr_history)
        except Exception as e: print('Error:', e)
        finally: load_model(self.model, 'tmp.h5')
    
    def cam(self, vx, final_layer, out_filters, classes):
        if not self.cam_model:
            self.cam_model = nn.Sequential(
                    *list(self.model.children())[:final_layer+1], 
                    nn.Conv2d(out_filters, classes, 1), 
                    AdaptiveConcatPool2d(1),
                    nn.Conv2d(classes*2, classes, 1),
                    Flatten(),
                ).cuda()
            self.cam_feature_layer = -4
            module_trainable(self.cam_model[:self.cam_feature_layer], False)
            #self.fit_onecycle()
            r = Runner(self.cam_model, self.train_loader, self.test_loader, 'adam', 'focal')
            for _ in range(5): r.fit_onecycle(1e-3)
            
        self.cam_feature = SaveFeatures(self.cam_model[self.cam_feature_layer])
        py = self.cam_model(vx.cuda())
        py = F.softmax(py, dim=-1)
        py = py.detach().cpu().numpy()[0]
        feat = self.cam_feature.features[0].detach().cpu().numpy()
        feat = np.maximum(0, feat)
        f2 = np.dot(np.rollaxis(feat,0,3), py)
        f2 -= f2.min()
        f2 /= f2.max()
        dx = vx.cpu().numpy().transpose(0,2,3,1)[0]
        import skimage
        plt.axis('off')
        plt.imshow(dx)
        ss = skimage.transform.resize(f2, dx.shape[:2])
        plt.imshow(ss, alpha=0.5, cmap='hot')
        module_trainable(self.model, True)
        self.cam_feature.remove()
