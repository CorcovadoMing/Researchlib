import os
from tqdm import tnrange
from ..utils import _register_method, plot_montage
from .history import History
from itertools import cycle
import torch
import random
from .liveplot import Liveplot

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
        data, target = augmentation_fn._forward(data, target, random.random(), random.random())

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
        
def _list_avg(l):
    return sum(l) / len(l) 

#================================================================================== 
            
            
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

            for m in metrics: m.reset()

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

            liveplot.record(epoch, 'norm', _list_avg(norm))
            if liveplot._gan:
                liveplot.record(epoch, 'g_lr', [i['lr'] for i in self.optimizer[1].param_groups][-1])
                liveplot.record(epoch, 'd_lr', [i['lr'] for i in self.optimizer[0].param_groups][-1])
                liveplot.record(epoch, 'train_g_loss', _list_avg(g_loss_history))
                liveplot.record(epoch, 'train_d_loss', _list_avg(d_loss_history))
            else:
                liveplot.record(epoch, 'lr', [i['lr'] for i in self.optimizer.param_groups][-1])
                liveplot.record(epoch, 'train_loss', _list_avg(loss_history))

            # Output metrics
            for m in metrics: matrix_records.add(m.output(), prefix='train')
                
            if liveplot._gan:
                loss_records = {'d_loss': _list_avg(d_loss_history), 'g_loss': _list_avg(g_loss_history)}
            else:
                loss_records = {'loss': _list_avg(loss_history)}

            self.history_.add(loss_records, prefix='train')
            self.history_ += matrix_records

            try:
                liveplot.record(epoch, 'train_acc', self.history_.records['train_acc'][-1])
            except:
                pass

            if liveplot._gan:
                liveplot.record(epoch, 'inception_score', self.history_.records['train_inception_score'][-1])
                liveplot.record(epoch, 'fid', self.history_.records['train_fid'][-1])

            for callback_func in callbacks:
                callback_func.on_epoch_end(model=self.model,
                                           train_loader=self.train_loader,
                                           optimizer=self.optimizer,
                                           epoch=epoch)

            if liveplot._gan:
                ema = self.ema > 0 and self.epoch > self.ema_start
                _gan_sample = self.model.sample(4,
                                                inference=True,
                                                gpu=True,
                                                ema=ema)
                _gan_sample = _gan_sample.detach().cpu().numpy().transpose(
                    (0, 2, 3, 1))
                _grid = plot_montage(_gan_sample, 2, 2, False)
                liveplot.record(epoch, 'image', _grid)
                    
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
                    metrics=metrics,
                    callbacks=callbacks,
                    inputs=self.inputs)

                self.history_.add(loss_records, prefix='val')
                self.history_ += matrix_records
                try:
                    liveplot.record(epoch, 'val_acc', self.history_.records['val_acc'][-1])
                except:
                    pass
                liveplot.record(epoch, 'val_loss', self.history_.records['val_loss'][-1])

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
