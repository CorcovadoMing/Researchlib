import os
from tqdm import tnrange
from ..utils import _register_method, plot_montage, _is_port_in_use, set_lr, inifinity_loop, Annealer
from .history import History
import torch
import random
from .liveplot import Liveplot
from .prefetch import *
import pickle
from ..frontend.dashboard import _Dashboard
from apex import amp
import numpy as np

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def set_policy(self, policy, lr, epochs):
    if self.__class__.__runner_settings__['optimizer'] == 'cocob':
        print(
            'Optimizer cocob no need to set the learning rate, disabled any learning rate settings (lr, annealing, etc.,)'
        )
    else:
        if policy == 'cyclical':
            try:
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=lr / 50,
                    max_lr=lr,
                    step_size_up=len(self.train_loader),
                    step_size_down=len(self.train_loader),
                    cycle_momentum=True)
            except:  # No momentum
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=lr / 50.,
                    max_lr=lr,
                    step_size_up=len(self.train_loader),
                    step_size_down=len(self.train_loader),
                    cycle_momentum=False)
        elif policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, epochs * len(self.train_loader))


@register_method
def fit(self,
        epochs,
        lr=1e-3,
        policy='cosine',
        warmup=5,
        mixup_alpha=0,
        metrics=[],
        callbacks=[],
        _id='none',
        self_iterative=False,
        iterations=0,
        multisteps=None,
        prefetch=False,
        plot=False):

    self.__class__.__fit_settings__[
        f'epoch_{self.epoch}-{self.epoch+epochs}'] = locals()

    self.set_optimizer()

    # Fix issue the dashboard is down while training is interrupted
    if not _is_port_in_use(8050):
        dash = _Dashboard(verbose=False)
        dash.start()

    if multisteps is not None:
        if policy == 'cosine':
            print(
                'Cosine annealing is not suitable to combine with multisteps annealing, disable multisteps automatically.'
            )
        else:
            assert type(multisteps) == list
            self.multisteps = multisteps

    self._fit(
        epochs,
        lr,
        mixup_alpha,
        metrics,
        callbacks,
        _id,
        self_iterative,
        iterations=iterations,
        policy=policy,
        plot=plot,
        prefetch=prefetch,
        warmup=warmup)


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
        data, target = [torch.from_numpy(i) for i in data
                       ], [torch.from_numpy(i) for i in target]
    return data, target


@register_method
def _process_data(self, data, target, mixup_alpha, inference):
    # On the flay preprocessing
    for preprocessing_fn in self.preprocessing_list:
        data, target = preprocessing_fn._forward(data, target)

    # On the fly augmentation
    if not inference:
        for augmentation_fn in self.augmentation_list:
            data, target = augmentation_fn._forward(data, target,
                                                    random.random(),
                                                    random.random())

    # Mixup
    _mixup_alpha = Annealer.get_trace('mixup_alpha') if mixup_alpha == Annealer else mixup_alpha
    if _mixup_alpha > 0 and not inference:
        lam = np.random.beta(_mixup_alpha, _mixup_alpha)
        target_res = []
        for i in range(len(data)):
            index = torch.randperm(data[i].size(0))
            data[i] = lam * data[i] + (1 - lam) * data[i][index]
            target_res.append(target[i][index])
    else:
        lam = None
        target_res = None
    return data, target, target_res, lam


@register_method
def _iteration_pipeline(self, loader, mixup_alpha, inference=False):
    for batch_idx, data_pack in inifinity_loop(loader):
        x, y = self._process_type(data_pack, self.inputs)
        x, y, y_res, lam = self._process_data(x, y, mixup_alpha, inference)
        yield x, y, y_res, lam


#==================================================================================


def _list_avg(l):
    return sum(l) / len(l)


#==================================================================================


@register_method
def _fit(self, epochs, lr, mixup_alpha, metrics, callbacks, _id, self_iterative,
         iterations, policy, warmup, prefetch, plot):

    base_lr = lr
    set_lr(self.optimizer, base_lr)

    if type(self.optimizer) == list:
        for i in self.optimizer:
            i.zero_grad()
    else:
        self.optimizer.zero_grad()

    liveplot = Liveplot(self.model, len(self.train_loader), plot)

    if iterations == 0:
        iterations = len(self.train_loader)

    if len(self.experiment_name) == 0:
        self.start_experiment('default')

    exist_experiments = pickle.loads(liveplot.redis.get('experiment'))
    if self.experiment_name not in exist_experiments:
        exist_experiments.append(self.experiment_name)
    liveplot.redis.set('experiment', pickle.dumps(exist_experiments))

    self.preload_gpu()

    self.model, self.optimizer = amp.initialize(
        self.model,
        self.optimizer,
        opt_level='O1',
        enabled=self.fp16,
        loss_scale=1)

    try:
        if len(self.default_metrics):
            metrics = self.default_metrics + metrics
            
        train_prefetcher = self._iteration_pipeline(self.train_loader, mixup_alpha)
        if prefetch:
            train_prefetcher = BackgroundGenerator(train_prefetcher)
        if self.test_loader:
            test_loader = self._iteration_pipeline(
                self.test_loader, mixup_alpha, inference=True)

        for epoch in range(1, epochs + 1):
            # Flat warmup implementation
            if epoch == (warmup + 1):
                self.set_policy(policy, base_lr, epochs - warmup)

            for callback_func in callbacks:
                callback_func.on_epoch_begin(
                    model=self.model,
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
                m.reset()

            iteration_break = iterations
            liveplot.redis.set('stage', 'train')
            liveplot.timer.clear()
            self.model.train()
            for batch_idx, (x, y, y_res, lam) in enumerate(train_prefetcher):
                liveplot.update_progressbar(batch_idx + 1)
                if self.is_cuda:
                    x, y = [i.cuda() for i in x], [i.cuda() for i in y]
                    if lam is not None:
                        # Mixup enabled
                        y_res = [i.cuda() for i in y_res]

                self.train_fn(
                    train=True,
                    model=self.model,
                    data=x,
                    target=y,
                    target_res=y_res,
                    lam=lam,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_fn,
                    reg_fn=self.reg_fn,
                    reg_weights=self.reg_weights,
                    epoch=self.epoch,
                    callbacks=callbacks,
                    metrics=metrics,
                    loss_history=loss_history,
                    g_loss_history=g_loss_history,
                    d_loss_history=d_loss_history,
                    norm=norm,
                    matrix_records=matrix_records,
                    bar=None)

                liveplot.update_loss_desc(self.epoch, g_loss_history,
                                          d_loss_history, loss_history)
                
                # Global iteration annealling
                Annealer._iteration_step()

                iteration_break -= 1
                if iteration_break == 0:
                    break

            self.model.eval()

            liveplot.record(epoch, 'norm', _list_avg(norm))
            if liveplot._gan:
                liveplot.record(epoch, 'g_lr', [
                    i['lr'] for i in self.optimizer[1].param_groups
                ][-1])
                liveplot.record(epoch, 'd_lr', [
                    i['lr'] for i in self.optimizer[0].param_groups
                ][-1])
                liveplot.record(epoch, 'train_g_loss',
                                _list_avg(g_loss_history))
                liveplot.record(epoch, 'train_d_loss',
                                _list_avg(d_loss_history))
            else:
                liveplot.record(epoch, 'lr',
                                [i['lr'] for i in self.optimizer.param_groups
                                ][-1])
                liveplot.record(epoch, 'train_loss', _list_avg(loss_history))

            # Output metrics
            for m in metrics:
                matrix_records.add(m.output(), prefix='train')

            if liveplot._gan:
                loss_records = {
                    'd_loss': _list_avg(d_loss_history),
                    'g_loss': _list_avg(g_loss_history)
                }
            else:
                loss_records = {'loss': _list_avg(loss_history)}

            self.history_.add(loss_records, prefix='train')
            self.history_ += matrix_records

            try:
                liveplot.record(epoch, 'train_acc',
                                self.history_.records['train_acc'][-1])
            except:
                pass

            if liveplot._gan:
                liveplot.record(
                    epoch, 'inception_score',
                    self.history_.records['train_inception_score'][-1])
                liveplot.record(epoch, 'fid',
                                self.history_.records['train_fid'][-1])

            for callback_func in callbacks:
                callback_func.on_epoch_end(
                    model=self.model,
                    train_loader=self.train_loader,
                    optimizer=self.optimizer,
                    epoch=epoch)

            if liveplot._gan:
                _gan_sample = self.model.sample(4, inference=True, gpu=True)
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

            liveplot.redis.set('stage', 'validate')

            if self.test_loader:
                loss_records, matrix_records = self.validate_fn(
                    model=self.model,
                    test_loader=test_loader,
                    loss_fn=self.loss_fn,
                    is_cuda=self.is_cuda,
                    epoch=self.epoch,
                    metrics=metrics,
                    callbacks=callbacks,
                    inputs=self.inputs)

                self.history_.add(loss_records, prefix='val')
                self.history_ += matrix_records
                try:
                    liveplot.record(epoch, 'val_acc',
                                    self.history_.records['val_acc'][-1])
                except:
                    pass
                liveplot.record(epoch, 'val_loss',
                                self.history_.records['val_loss'][-1])

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
            if critic is not None and self.monitor_mode(critic,
                                                        self.monitor) == critic:
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

            # Steps Anneling
            if self.epoch in self.multisteps:
                base_lr *= 0.1
                if policy == 'cyclical':
                    # Strange trick, don't change this line for cyclical
                    set_lr(self.optimizer, base_lr / 50, 'initial_lr')
                else:
                    set_lr(self.optimizer, base_lr)
                self.set_policy(policy, base_lr)

            # Self-interative
            if self_iterative:
                with torch.no_grad():
                    for i in tnrange(len(self.train_loader.dataset.tensors[0])):
                        self.train_loader.dataset.tensors[1][i] = \
                        self.model(self.train_loader.dataset.tensors[0][i].unsqueeze(0).cuda()).detach().cpu()[0]
                        torch.cuda.empty_cache()
            
            # Global epoch annealling
            Annealer._epoch_step()

        liveplot.redis.set('stage', 'stop')

    except:
        raise

    finally:
        self.unload_gpu()
        _STOP_GPU_MONITOR_ = True
