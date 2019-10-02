import os
from tqdm import tnrange
from ..utils import _register_method, plot_montage, _is_port_in_use, set_lr, inifinity_loop, Annealer, ParameterManager, update_optim
from ..layers import layer
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
def _iteration_pipeline(self, loader):
    for batch_idx, (x, y) in inifinity_loop(loader):
        if type(x) != torch.Tensor:
            x = torch.from_numpy(x)
        if type(y) != torch.Tensor:
            y = torch.from_numpy(y)
        yield x, y


def _anneal_policy(anneal_type):
    if anneal_type == 'cosine':
        anneal_policy = Annealer.Cosine
    elif anneal_type == 'linear':
        anneal_policy = Annealer.Linear
    else:
        anneal_policy = Annealer.Fixed
    return anneal_policy


@register_method
def fit(
    self,
    epochs,
    lr = 3e-3,
    policy = 'linear',
    warmup = 5,
    warmup_policy = 'linear',
    metrics = [],
    callbacks = [],
    _id = 'none',
    self_iterative = False,
    iterations = 0,
    multisteps = [],
    prefetch = True,
    plot = False,
    init_optim = True,
    **kwargs
):
    
    self.__class__.__fit_settings__[f'epoch_{self.epoch}-{self.epoch+epochs-1}'] = locals()
    
    parameter_manager = ParameterManager(**kwargs)
    

    # ----------------------------------------------
    # Dashboard
    # ----------------------------------------------
    if not _is_port_in_use(8050):
        dash = _Dashboard(verbose = False)
        dash.start()

    parameter_manager = ParameterManager(**kwargs)
    
    
    # ----------------------------------------------
    # Setting loaders
    # ----------------------------------------------
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    buffered_epochs = epochs + 1
    train_loader = self.train_loader.get_generator(batch_size, epochs=buffered_epochs)
    self.train_loader_length = len(train_loader)
    liveplot = Liveplot(self.train_loader_length, plot)
    train_loader = self._iteration_pipeline(train_loader)
    train_loader = BackgroundGenerator(train_loader) if prefetch else train_loader
    if self.test_loader:
        test_loader = self.test_loader.get_generator(batch_size, epochs=buffered_epochs)
        self.test_loader_length = len(test_loader)
        test_loader = self._iteration_pipeline(test_loader)
        test_loader = BackgroundGenerator(test_loader) if prefetch else test_loader
        
    if iterations == 0:
        iterations = self.train_loader_length
        
        
    # ----------------------------------------------
    # Setting experiments
    # ----------------------------------------------
    if len(self.experiment_name) == 0:
        self.start_experiment('default')
        
    exist_experiments = pickle.loads(liveplot.redis.get('experiment'))
    if self.experiment_name not in exist_experiments:
        exist_experiments.append(self.experiment_name)
        
    liveplot.redis.set('experiment', pickle.dumps(exist_experiments))
    
    
    
    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------
    if len(self.default_metrics):
        metrics = self.default_metrics + metrics
    
    
    # ----------------------------------------------
    # MISC
    # ----------------------------------------------
    fixed_mmixup = parameter_manager.get_param('fixed_mmixup', validator = lambda x: type(x) == list)
    random_mmixup = parameter_manager.get_param('random_mmixup', validator = lambda x: len(x) == 2 and type(x) == list)
    mmixup_alpha = parameter_manager.get_param('mmixup_alpha', validator = lambda x: type(x) == float)
    
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Initialization should be completed before this line
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    self.preload_gpu()
    
    
    # ----------------------------------------------
    # optimizers, (LR, warmup, weight_decay, etc.,)
    # ----------------------------------------------
    if init_optim:
        self.set_optimizer()
    
    self.multisteps = multisteps

    # Weight decay
    weight_decay = parameter_manager.get_param('weight_decay', 1)
    if weight_decay > 0:
        weight_decay_policy = parameter_manager.get_param('weight_decay_policy', 'cosine')
        Annealer.set_trace('weight_decay', epochs * iterations, [0, weight_decay], 'iteration',_anneal_policy(weight_decay_policy))
        Annealer._iteration_step(key = 'weight_decay')
        
    # Warmup
    warmup = max(0, warmup)
    if warmup > 0:
        Annealer.set_trace('warmup_lr', warmup * iterations, [0, lr], 'iteration', _anneal_policy(warmup_policy))
        Annealer._iteration_step(key = 'warmup_lr')
    
    # FP16
    fp16 = parameter_manager.get_param('fp16', False)
    loss_scale = parameter_manager.get_param('loss_scale', 1)
    self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level = 'O1', enabled = fp16, loss_scale = loss_scale)
    
    
    # ----------------------------------------------
    # Final verification
    # ----------------------------------------------
    ParameterManager.verify_kwargs(**kwargs)
    
    try:
        for epoch in range(1, epochs+1):
            
            # ----------------------------------------------
            # Pre-config
            # ----------------------------------------------
            if epoch == (warmup + 1):
                Annealer.set_trace('regular_lr', (epochs - warmup) * iterations, [lr, 0], 'iteration', _anneal_policy(policy))
                Annealer._iteration_step(key = 'regular_lr')
                
            # Set LR
            if epoch <= warmup:
                cur_lr = Annealer.get_trace('warmup_lr')
            else:
                cur_lr = Annealer.get_trace('regular_lr')
            set_lr(self.optimizer, cur_lr)
            
            # Set weight decay
            if weight_decay > 0:
                weight_decay = Annealer.get_trace('weight_decay')
                update_optim(self.optimizer, weight_decay, key = 'weight_decay')
            
            for callback_func in callbacks:
                callback_func.on_epoch_begin(
                    model = self.model,
                    train_loader = self.train_loader,
                    optimizer = self.optimizer,
                    epoch = self.epoch
                )

            
            # ----------------------------------------------
            # Training stage
            # ----------------------------------------------
            
            liveplot.redis.set('stage', 'train')
            liveplot.timer.clear()            
            self.model.train()
            
            for m in metrics:
                m.reset()
            
            loss_record = 0
            norm_record = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if mmixup_alpha is not None:
                    batch_size = inputs[0].size(0)
                    if fixed_mmixup is None and random_mmixup is None:
                        random_mmixup = [0, layer.ManifoldMixup.block_counter]
                    lam = layer.ManifoldMixup.setup_batch(mmixup_alpha, batch_size, fixed_mmixup, random_mmixup)
                    targets, targets_res = layer.ManifoldMixup.get_y(targets)
                    targets, targets_res = targets.cuda(), targets_res.cuda()
                else:
                    targets_res = None
                    lam = None
                
                
                inputs, targets = inputs.cuda(), targets.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn[0](outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                # May be a bottleneck for GPU utilization
                for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
                    norm_record += p.grad.data.norm(2).item() ** 2
    
                for m in metrics:
                    m.forward([outputs, targets])
                    
                loss_record += loss.item()
                # May be a bottleneck for GPU utilization
                liveplot.update_progressbar(batch_idx + 1)
                liveplot.update_desc(epoch, loss_record / (batch_idx + 1), metrics, self.monitor)

                if batch_idx == (self.train_loader_length-1):
                    break
                
                Annealer._iteration_step()

            loss_record = loss_record / (batch_idx + 1)
            norm_record = (norm_record ** 0.5) / (batch_idx + 1)
            liveplot.record(epoch, 'lr', [i['lr'] for i in self.optimizer.param_groups][-1])
            liveplot.record(epoch, 'train_loss', loss_record)
            liveplot.record(epoch, 'norm', norm_record)
            self.history_.add({'loss': loss_record}, prefix = 'train')
            self.history_.record_matrix(metrics, prefix = 'train')
            try:
                liveplot.record(epoch, 'train_acc', self.history_.records['train_acc'][-1])
            except:
                pass
                
            # ----------------------------------------------
            # Validation stage
            # ----------------------------------------------
            liveplot.redis.set('stage', 'validate')
            loss_record = self.validate_fn(test_loader, metrics)
            liveplot.record(epoch, 'val_loss', loss_record)
            self.history_.add({'loss': loss_record / (batch_idx + 1)}, prefix = 'val')
            self.history_.record_matrix(metrics, prefix = 'val')
            try:
                liveplot.record(epoch, 'val_acc', self.history_.records['val_acc'][-1])
            except:
                pass
                        
            # ----------------------------------------------
            # Check point
            # ----------------------------------------------
            epoch_str = str(self.epoch)
            monitor_target = 'val_' + self.monitor_state if self.test_loader else 'train_' + self.monitor_state
            if monitor_target in self.history_.records:
                critic = self.history_.records[monitor_target][-1]
            else:
                critic = None

            # Checkpoint
            checkpoint_model_name = os.path.join(self.checkpoint_path, 'checkpoint_' + _id + '_epoch_' + str(self.epoch))
            self.save(checkpoint_model_name)
            if critic is not None and self.monitor_mode(critic, self.monitor) == critic:
                self.monitor = critic
                best_checkpoint_model_name = os.path.join(self.checkpoint_path, 'best_' + _id)
                self.save(best_checkpoint_model_name)
                epoch_str += '*'

            
            # ----------------------------------------------
            # Post-config
            # ----------------------------------------------
            
            liveplot.plot(self.epoch, self.history_, epoch_str)
            
            for callback_func in callbacks:
                callback_func.on_epoch_end(
                    model = self.model,
                    train_loader = self.train_loader,
                    optimizer = self.optimizer,
                    epoch = self.epoch
                )
            
            # Steps anneling
            if self.epoch in self.multisteps:
                srange = Annealer.get_srange('regular_lr')
                srange = [i * 0.1 for i in srange]
                Annealer.update_attr('regular_lr', 'srange', srange)
                
            # Global epoch annealing
            Annealer._epoch_step()
            
            # Finish this epoch
            self.epoch += 1
            
        
    except:
        raise

    finally:
        self.unload_gpu()
        _STOP_GPU_MONITOR_ = True
        liveplot.redis.set('stage', 'stop')