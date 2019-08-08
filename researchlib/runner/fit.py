import os
from tqdm import tnrange
from ..utils import _register_method, _get_iteration, set_lr, plot_montage
from .history import History
from ..models import GANModel
from itertools import cycle
from ipywidgets import IntProgress, Output, HBox, Label
import hiddenlayer as hl
import threading
import torch
from pynvml import *
import time
import random

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def fit_iteration(self,
                  iteration,
                  lr=1e-3,
                  policy='cyclical',
                  augmentor=None,
                  mixup_alpha=0,
                  metrics=[],
                  callbacks=[],
                  _id='none',
                  self_iterative=False):
    _, callbacks = self._set_policy(policy, lr, callbacks)
    self._fit(1,
              lr,
              augmentor,
              mixup_alpha,
              metrics,
              callbacks,
              _id,
              self_iterative,
              cycle=True,
              total=iteration)





@register_method
def fit_xy(self,
           data_pack,
           inputs,
           lr=1e-3,
           policy='cyclical',
           augmentor=None,
           mixup_alpha=0,
           metrics=[],
           callbacks=[],
           _id='none',
           self_iterative=False,
           _auto_gpu=False,
           _train=True):
    
    _, callbacks = self._set_policy(policy, lr, callbacks)
    if len(self.experiment_name) == 0:
        self.start_experiment('default')

    if _auto_gpu:
        self.preload_gpu()

    try:
        self._fit_xy(data_pack,
                     inputs,
                     augmentor,
                     mixup_alpha,
                     callbacks,
                     metrics,
                     loss_history=[],
                     g_loss_history=[],
                     d_loss_history=[],
                     norm=[],
                     matrix_records=History(),
                     bar=None,
                     train=_train)
    except:
        raise
    finally:
        if _auto_gpu:
            self.unload_gpu()


@register_method
def fit(self,
        epochs,
        lr=1e-3,
        policy='cyclical',
        augmentor=None,
        mixup_alpha=0,
        metrics=[],
        callbacks=[],
        _id='none',
        self_iterative=False,
        accum_gradient=1,
        accum_freq=100):
    
    total_epochs, callbacks = self._set_policy(policy, lr, callbacks, epochs)

    self._accum_gradient = 1
    self._accum_freq = accum_freq
    self._accum_target_gradient = accum_gradient
    self._accum_step = False
    self._accum_current = 0
    self._fit(total_epochs,
              lr,
              augmentor,
              mixup_alpha,
              metrics,
              callbacks,
              _id,
              self_iterative,
              cycle=False)


@register_method
def _process_type(self, data_pack, inputs):
    # DALI
    if type(data_pack[0]) == dict:
        data, target = data_pack[0]['data'], data_pack[0]['label']
    else:
        data, target = data_pack[:inputs], data_pack[inputs:]

    if type(data) != list and type(data) != tuple: 
        data = [data]
    if type(target) != list and type(target) != tuple: 
        target = [target]
    if type(data[0]) != torch.Tensor:
        data, target = [torch.from_numpy(i) for i in data], [torch.from_numpy(i) for i in target]
    return data, target


@register_method
def _process_data(self, data, target, augmentor, mixup_alpha):
    def mixup_loss_fn(loss_fn, x, y, y_res, lam):
        return lam * loss_fn(x, y) + (1 - lam) * loss_fn(x, y_res)

    # On the flay preprocessing
    for preprocessing_fn in self.preprocessing_list:
        data, target = preprocessing_fn._forward(data, target)

    # On the fly augmentation
    for augmentation_fn in self.augmentation_list:
        data, target = augmentation_fn._forward(data, target, 0.5, random.random())

    # Target type refine (should be remove after refined loss function)
    while len(target) != len(self.require_long_):
        self.require_long_.append(self.require_long_[-1])
    target = [i.long() if j else i for i, j in zip(target, self.require_long_)]

    # GPU
    if self.is_cuda:
        data, target = [i.cuda() for i in data], [i.cuda() for i in target]

    # Mixup
    if mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        index = torch.randperm(data[0].size(0))
        data[0] = lam * data[0] + (1 - lam) * data[0][index]
        target_res = [i[index] for i in target]
        if self.is_cuda: target_res = [i.cuda() for i in target_res]
        self.lam = lam
        self.mixup_loss_fn = mixup_loss_fn
    else:
        self.lam = None
        self.mixup_loss_fn = None
        target_res = None
    return data, target, target_res


@register_method
def _fit_xy(self, data_pack, inputs, augmentor, mixup_alpha, callbacks,
            metrics, loss_history, g_loss_history, d_loss_history, norm,
            matrix_records, bar, train):
    self.data, self.target = self._process_type(data_pack, inputs)
    self.data, self.target, self.target_res = self._process_data(
        self.data, self.target, augmentor, mixup_alpha)

    self.model.train()
    self.train_fn(train=train,
                  model=self.model,
                  data=self.data,
                  target=self.target,
                  target_res=self.target_res,
                  optimizer=self.optimizer,
                  loss_fn=self.loss_fn,
                  mixup_loss_fn=self.mixup_loss_fn,
                  reg_fn=self.reg_fn,
                  reg_weights=self.reg_weights,
                  epoch=self.epoch,
                  keep_x_shape=self.keep_x_shape_,
                  keep_y_shape=self.keep_y_shape_,
                  mixup_alpha=mixup_alpha,
                  callbacks=callbacks,
                  metrics=metrics,
                  loss_history=loss_history,
                  g_loss_history=g_loss_history,
                  d_loss_history=d_loss_history,
                  norm=norm,
                  matrix_records=matrix_records,
                  bar=bar)
    self.model.eval()

#==================================================================================
    
def _cycle(data, _cycle_flag=False):
    if _cycle_flag:
        return cycle(enumerate(data))
    else:
        return enumerate(data)

def _get_gpu_monitor():
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    s = nvmlDeviceGetUtilizationRates(handle)
    return int(100 * (info.used / info.total)), s.gpu

def _gpu_monitor_worker(membar, utilsbar):
    while True:
        global _STOP_GPU_MONITOR_
        if _STOP_GPU_MONITOR_:
            membar.value, utilsbar.value = 0, 0
            membar.description, utilsbar.description = 'M', 'U'
            break
        m, u = _get_gpu_monitor()
        membar.value, utilsbar.value = m, u
        membar.description, utilsbar.description = 'M: ' + str(
            m) + '%', 'U: ' + str(u) + '%'
        time.sleep(0.2)
        
def _list_avg(l):
    return sum(l) / len(l) 

#================================================================================== 

class Liveplot:
    def __init__(self, model, total_iteration):
        self._gan = True if type(model) == GANModel else False
        self.history = hl.History()
        
        # Label + Pregress
        self.progress = Output()
        self.progress_label = Output()
        display(HBox([self.progress_label, self.progress]))
        with self.progress:
            self.progress_bar = IntProgress(bar_style='info')
            self.progress_bar.min = 0
            self.progress_bar.max = total_iteration
            display(self.progress_bar)
        with self.progress_label:
            self.progress_label_text = Label(value="Initialization")
            display(self.progress_label_text)
        
        # 4 chartplots
        self.loss_plot = Output()
        self.matrix_plot = Output()
        display(HBox([self.loss_plot, self.matrix_plot]))
        self.lr_plot = Output()
        self.norm_plot = Output()
        display(HBox([self.lr_plot, self.norm_plot]))
        
        # GAN visualization
        if self._gan:
            self.gan_gallary = Output()
            display(self.gan_gallary)
            self.gan_canvas = hl.Canvas()
        
        # Memory and Log
        self.gpu_mem_monitor = Output()
        self.gpu_utils_monitor = Output()
        self.text_log = Output()
        display(HBox([self.gpu_mem_monitor, self.gpu_utils_monitor, self.text_log]))
        
        with self.gpu_mem_monitor:
            self.gpu_mem_monitor_bar = IntProgress(orientation='vertical', bar_style='success')
            self.gpu_mem_monitor_bar.description = 'M: 0%'
            self.gpu_mem_monitor_bar.min = 0
            self.gpu_mem_monitor_bar.max = 100
            display(self.gpu_mem_monitor_bar)
            
        with self.gpu_utils_monitor:
            self.gpu_utils_monitor_bar = IntProgress(orientation='vertical', bar_style='success')
            self.gpu_utils_monitor_bar.description = 'U: 0%'
            self.gpu_utils_monitor_bar.min = 0
            self.gpu_utils_monitor_bar.max = 100
            display(self.gpu_utils_monitor_bar)
        
        # Canvas
        self.loss_canvas = hl.Canvas()
        self.matrix_canvas = hl.Canvas()
        self.lr_canvas = hl.Canvas()
        self.norm_canvas = hl.Canvas()
        
        # Start monitor thread
        global _STOP_GPU_MONITOR_
        _STOP_GPU_MONITOR_ = False
        self.thread = threading.Thread(target=_gpu_monitor_worker,
                                      args=(self.gpu_mem_monitor_bar, self.gpu_utils_monitor_bar))
        self.thread.start()
    
    
    def update_progressbar(self, value):
        self.progress_bar.value = value
    
    
    def update_loss_desc(self, epoch, g_loss_history, d_loss_history, loss_history):
        if self._gan:
            self.progress_label_text.value = f'Epoch: {epoch}, G Loss: {_list_avg(g_loss_history):.4f}, D Loss: {_list_avg(d_loss_history):.4f}'
                                 #Iter: ({batch_idx+1}/{total_iteration}:{desc_current}/{self._accum_gradient})'
        else:
            self.progress_label_text.value = f'Epoch: {epoch}, Loss: {_list_avg(loss_history):.4f}'

    
    def plot(self, epoch, history_, epoch_str):
        if self._gan:
            with self.gan_gallary:
                self.gan_canvas.draw_image(self.history['image'])
            
        with self.loss_plot:
            if self._gan:
                self.loss_canvas.draw_plot([self.history["train_g_loss"], self.history['train_d_loss']])
            else:
                self.loss_canvas.draw_plot([self.history["train_loss"], self.history['val_loss']])
        with self.matrix_plot:
            if self._gan:
                self.matrix_canvas.draw_plot([self.history['inception_score'], self.history['fid']])
            else:
                self.matrix_canvas.draw_plot([self.history['train_acc'], self.history['val_acc']])
        with self.lr_plot:
            if self._gan:
                self.lr_canvas.draw_plot([self.history['g_lr'], self.history['d_lr']])
            else:
                self.lr_canvas.draw_plot([self.history['lr']])
        with self.norm_plot:
            self.norm_canvas.draw_plot([self.history['norm']])

        with self.text_log:
            state = []
            fs = '{:^14}'
            if epoch == 1:
                print(('{:^10}' +
                       (fs *
                        (len(history_.records.keys()) - 1))).format(
                            'Epochs',
                            *list(history_.records.keys())[:-1]))
                print('================================================================')
            state.append('{:^10}'.format(epoch_str))
            for i in history_.records:
                if i != 'saved':
                    state.append('{:^14.4f}'.format(history_.records[i][-1]))
            print(''.join(state))
            
            
@register_method
def _fit(self,
         epochs,
         lr,
         augmentor,
         mixup_alpha,
         metrics,
         callbacks,
         _id,
         self_iterative,
         cycle,
         total=0):
    
    _gan = True if type(self.model) == GANModel else False
    
    if type(self.optimizer) == list:
        for i in self.optimizer:
            i.zero_grad()
    else:
        self.optimizer.zero_grad()

    liveplot = Liveplot(self.model, len(self.train_loader))

    if total == 0:
        total = len(self.train_loader)

    if len(self.experiment_name) == 0:
        self.start_experiment('default')

    self.preload_gpu()

    try:
        if len(self.default_metrics):
            metrics = self.default_metrics + metrics

        for epoch in range(1, epochs + 1):
            #             if epoch % self._accum_freq == 0:
            #                 self._accum_gradient = min(self._accum_target_gradient, self._accum_gradient*2)
            self._accum_gradient = self._accum_target_gradient

            for callback_func in callbacks:
                callback_func.on_epoch_begin(model=self.model,
                                             train_loader=self.train_loader,
                                             optimizer=self.optimizer,
                                             epoch=self.epoch)

            loss_history = []
            g_loss_history = []
            d_loss_history = []
            norm = []
            inception_score = []
            matrix_records = History()

            for m in metrics:
                try:
                    m.reset()
                except:
                    pass

            iteration_break = total
            for batch_idx, data_pack in _cycle(self.train_loader, cycle):
                self._accum_current += 1
                desc_current = self._accum_current
                if self._accum_current == self._accum_gradient:
                    self._accum_current = 0
                    self._accum_step = True
                else:
                    self._accum_step = False

                liveplot.update_progressbar(batch_idx+1)
                
                self._fit_xy(data_pack,
                             self.inputs,
                             augmentor,
                             mixup_alpha,
                             callbacks,
                             metrics,
                             loss_history,
                             g_loss_history,
                             d_loss_history,
                             norm,
                             matrix_records,
                             None,
                             train=True)

                liveplot.update_loss_desc(self.epoch, g_loss_history, d_loss_history, loss_history)
    
                iteration_break -= 1
                if iteration_break == 0:
                    break

            liveplot.history.log(epoch, norm=_list_avg(norm))
            if _gan:
                liveplot.history.log(epoch, g_lr=[i['lr'] for i in self.optimizer[1].param_groups][-1])
                liveplot.history.log(epoch, d_lr=[i['lr'] for i in self.optimizer[0].param_groups][-1])
                liveplot.history.log(epoch, train_g_loss=_list_avg(g_loss_history))
                liveplot.history.log(epoch, train_d_loss=_list_avg(d_loss_history))
            else:
                liveplot.history.log(epoch, lr=[i['lr'] for i in self.optimizer.param_groups][-1])
                liveplot.history.log(epoch, train_loss=_list_avg(loss_history))

            # Output metrics
            for m in metrics:
                try:
                    matrix_records.add(m.output(), prefix='train')
                except:
                    pass
                
            if _gan:
                loss_records = {'d_loss': _list_avg(d_loss_history), 'g_loss': _list_avg(g_loss_history)}
            else:
                loss_records = {'loss': _list_avg(loss_history)}

            self.history_.add(loss_records, prefix='train')
            self.history_ += matrix_records

            try:
                liveplot.history.log(epoch, train_acc=self.history_.records['train_acc'][-1])
            except:
                pass

            if _gan:
                liveplot.history.log(epoch, inception_score=self.history_.records['train_inception_score'][-1])
                liveplot.history.log(epoch, fid=self.history_.records['train_fid'][-1])

            for callback_func in callbacks:
                callback_func.on_epoch_end(model=self.model,
                                           train_loader=self.train_loader,
                                           optimizer=self.optimizer,
                                           epoch=epoch)

            if _gan:
                ema = self.ema > 0 and self.epoch > self.ema_start
                _gan_sample = self.model.sample(4,
                                                inference=True,
                                                gpu=True,
                                                ema=ema)
                _gan_sample = _gan_sample.detach().cpu().numpy().transpose(
                    (0, 2, 3, 1))
                _grid = plot_montage(_gan_sample, 2, 2, False)
                liveplot.history.log(epoch, image=_grid)
                #with gan_out:
                #    gan_canvas.draw_image(liveplot.history['image'])

                    
                    
            # SWA
            if self.swa and self.epoch >= self.swa_start and self.epoch % 2 == 0:
                if type(self.optimizer) == list:
                    for i in self.optimizer:
                        i.update_swa()
                else:
                    self.optimizer.update_swa()
                    
            if self.test_loader:
                loss_records, matrix_records = self.validate_fn(
                    model=self.model,
                    test_loader=self.test_loader,
                    loss_fn=self.loss_fn,
                    is_cuda=self.is_cuda,
                    epoch=self.epoch,
                    require_long=self.require_long_,
                    keep_x_shape=self.keep_x_shape_,
                    keep_y_shape=self.keep_y_shape_,
                    metrics=metrics,
                    callbacks=callbacks,
                    inputs=self.inputs)

                self.history_.add(loss_records, prefix='val')
                self.history_ += matrix_records
                try:
                    liveplot.history.log(epoch,
                                val_acc=self.history_.records['val_acc'][-1])
                except:
                    pass
                liveplot.history.log(epoch,
                            val_loss=self.history_.records['val_loss'][-1])

            epoch_str = str(self.epoch)

            monitor_target = 'val_' + self.monitor_state if self.test_loader else 'train_' + self.monitor_state
            if monitor_target in self.history_.records:
                critic = self.history_.records[monitor_target][-1]
            else:
                critic = None

            # Checkpoint
            checkpoint_model_name = os.path.join(
                self.checkpoint_path,
                'checkpoint_' + _id + '_epoch_' + str(self.epoch))
            self.save(checkpoint_model_name)
            if critic is not None and self.monitor_mode(
                    critic, self.monitor) == critic:
                self.monitor = critic
                best_checkpoint_model_name = os.path.join(
                    self.checkpoint_path, 'best_' + _id)
                self.save(best_checkpoint_model_name)
                epoch_str += '*'
                self.history_.add({'saved': '*'})
            else:
                self.history_.add({'saved': ''})

            liveplot.plot(self.epoch, self.history_, epoch_str)
            self.epoch += 1

            # Self-interative
            if self_iterative:
                with torch.no_grad():
                    for i in tnrange(len(
                            self.train_loader.dataset.tensors[0])):
                        self.train_loader.dataset.tensors[1][i] = \
                        self.model(self.train_loader.dataset.tensors[0][i].unsqueeze(0).cuda()).detach().cpu()[0]
                        torch.cuda.empty_cache()
    except:
        raise

    finally:
        self.preprocessing_list = []
        self.postprocessing_list = []
        self.augmentation_list = []
        self.unload_gpu()
        _STOP_GPU_MONITOR_ = True
