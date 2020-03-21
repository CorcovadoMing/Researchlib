from ..utils import _register_method, plot_montage, _is_port_in_use, Annealer, ParameterManager, update_optim
from .liveplot import Liveplot
from ..frontend.dashboard import _Dashboard
from ..ops import op
from ..loss import Loss
import numpy as np
import copy
import pickle
import os
import torch
import math


__methods__ = []
register_method = _register_method(__methods__)


def _anneal_policy(anneal_type):
    if anneal_type == 'cosine':
        anneal_policy = Annealer.Cosine
    elif anneal_type == 'linear':
        anneal_policy = Annealer.Linear
    elif anneal_type == 'poly2':
        anneal_policy = Annealer.Poly2
    else:
        anneal_policy = Annealer.Fixed
    return anneal_policy
    
    
    
def _clear_output(m):
    try:
        del m.outputs
    except:
        pass

def _clear_source(m):
    try:
        m.clear_source(True)
    except:
        pass

    
def _to_half(m):
        if isinstance(m, torch.nn.Module) and not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.half()
            
def _fix(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.float()
    if type(m) == Loss.AdaptiveRobust:
        m.float()
    
    
@register_method
def fit(
    self,
    epochs,
    lr = 3e-3,
    policy = 'linear',
    warmup = 5,
    warmup_policy = 'linear',
    expected_epochs = None,
    flatten = 0,
    flatten_lr = 0,
    final_anneal = 0,
    _id = 'none',
    iterations = 0,
    multisteps = [],
    plot = False,
    use_simple_liveplot = False,
    enable_liveplot = True,
    clear_data = True,
    opt_info = True,
    data_info = True,
    init = None,
    same_init = False,
    freeze = {},
    save_checkpoint = True,
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
        
    
    # ----------------------------------------------
    # Setting loaders
    # ----------------------------------------------
    
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    buffered_epochs = epochs + 1
    
    for k, v in self.model.graph.items():
        if type(v[0]) == op.Source:
            v[0].prepare_generator(buffered_epochs)
            self.train_loader_length = math.ceil(v[0].train_source_generator.__len__() / batch_size)
            if v[0].val_source is not None:
                self.test_loader_length = math.ceil(v[0].val_source_generator.__len__() / batch_size)
            else:
                self.test_loader_length = None
        if type(v[0]) == op.Generator:
            v[0].prepare_state(fp16, batch_size, data_info)
    
    inner_epochs = parameter_manager.get_param('inner_epochs', 1)
    if iterations == 0:
        try:
            # Classification
            iterations = self.train_loader_length
        except:
            # RL simulator
            iterations = 1
            self.train_loader_length = 1
            self.test_loader_length = None
    
    if enable_liveplot:
        liveplot = Liveplot(self.train_loader_length, self.test_loader_length, plot, use_simple_liveplot)
    else:
        liveplot = None
    
    # ----------------------------------------------
    # MISC
    # ----------------------------------------------
    
    Annealer._post_config(epochs, iterations)
    
    lars = parameter_manager.get_param('lars', False)
    
    if init is not None:
        self.init_model(init)
        
        if fp16:
            self.model.apply(_to_half)
            self.model.apply(_fix)
        
        self.set_optimizer(lars, opt_info)
        self.epoch = 1
    
        if same_init:
            init_model = os.path.join(self.checkpoint_path, 'init_model_' + _id)
            if os.path.exists(str(init_model) + '.model.pt'):
                self.load(init_model)
            else:
                self.save(init_model)
            self.set_optimizer(lars, opt_info)
            self.epoch = 1
    else:
        if self.optimizer is None:
            self.set_optimizer(lars, opt_info)
    
    fixed_mmixup = parameter_manager.get_param('fixed_mmixup', validator = lambda x: type(x) == list)
    random_mmixup = parameter_manager.get_param('random_mmixup', validator = lambda x: len(x) == 2 and type(x) == list)
    mmixup_alpha = parameter_manager.get_param('mmixup_alpha', validator = lambda x: type(x) == float)
        
    # ----------------------------------------------
    # Setting experiments
    # ----------------------------------------------
    if len(self.experiment_name) == 0:
        self.start_experiment('default')
    
    if liveplot is not None:
        exist_experiments = pickle.loads(liveplot.redis.get('experiment'))
        if self.experiment_name not in exist_experiments:
            exist_experiments.append(self.experiment_name)
        liveplot.redis.set('experiment', pickle.dumps(exist_experiments))
    
    self.history.start_logfile(self.checkpoint_path)
    
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Initialization should be completed before this line
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    self.preload_gpu()
    
    
    # ----------------------------------------------
    # optimizers, (LR, warmup, weight_decay, etc.,)
    # ----------------------------------------------
    grad_clip = parameter_manager.get_param('grad_clip', 9e9)
    accum_grad = parameter_manager.get_param('accum_grad', 1)
    self.accum_idx = 0
    
    if expected_epochs is None:
        expected_epochs = epochs
    self.multisteps = [int(i * expected_epochs) if i < 1 else int(i) for i in multisteps]
    step_decay = parameter_manager.get_param('step_decay', 0.1)

    # Weight decay
    weight_decay = parameter_manager.get_param('weight_decay', 5e-4)
    weight_decay_bias = parameter_manager.get_param('weight_decay_bias', True)
    if weight_decay > 0:
        weight_decay_policy = parameter_manager.get_param('weight_decay_policy', 'fixed')
        flag = Annealer.set_trace('weight_decay', epochs * iterations, [0, weight_decay], 'iteration', _anneal_policy(weight_decay_policy))
        if flag:
            Annealer._iteration_step(key = 'weight_decay')
        
    # Warmup
    warmup = max(0, warmup)
    if warmup > 0:
        flag = Annealer.set_trace('lr', warmup * iterations, [0, lr], 'iteration', _anneal_policy(warmup_policy))
        if flag:
            Annealer._iteration_step(key = 'lr')
    
    # For convergence
    bias_scale = parameter_manager.get_param('bias_scale', 1)
    
    # EMA
    ema_freq = max(0, parameter_manager.get_param('ema_freq', 0))
    ema_momentum = parameter_manager.get_param('ema_momentum', 0.99)
    ema = True if ema_freq > 0 else False
    if self.val_model is None:
        self.val_model = copy.deepcopy(self.model) if ema else self.model
    
    # ----------------------------------------------
    # Final verification
    # ----------------------------------------------
    ParameterManager.verify_kwargs(**kwargs)
    
    try:
        for epoch in range(1, epochs + 1):
            
            # ----------------------------------------------
            # Pre-config
            # ----------------------------------------------
            self.model.train()
            
            # Adjust freeze schedule
            for module in freeze:
                start, end, freeze_bn = freeze[module]
                if epoch >= start and epoch < end:
                    if freeze_bn:
                        module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
                else:
                    for p in module.parameters():
                        p.requires_grad = True
            
            
            # Adjust lr schedule
            if epoch == (warmup + 1):
                Annealer.set_trace('lr', (epochs - warmup - flatten) * iterations, [lr, flatten_lr], 'iteration', _anneal_policy(policy))
                Annealer._iteration_step(key = 'lr')
            
            if epoch == (epochs - flatten + 1):
                Annealer.set_trace('lr', flatten * iterations, [flatten_lr, flatten_lr], 'iteration', _anneal_policy('fixed'))
                Annealer._iteration_step(key = 'lr')
                
            if epoch == (epochs - final_anneal + 1):
                Annealer.set_trace('lr', final_anneal * iterations, [lr, 0], 'iteration', _anneal_policy('linear'))
                

            # ----------------------------------------------
            # Training stage
            # ----------------------------------------------
            if liveplot is not None:
                liveplot.redis.set('stage', 'train')
                liveplot.train_timer.clear()
            # Training function
            loss_record, norm_record, metrics_record = self.train_fn(liveplot=liveplot,
                                                                     batch_size=batch_size,
                                                                     mmixup_alpha=mmixup_alpha, 
                                                                     fixed_mmixup=fixed_mmixup, 
                                                                     random_mmixup=random_mmixup,
                                                                     epoch=epoch,
                                                                     inner_epochs = inner_epochs,
                                                                     warmup=warmup,
                                                                     weight_decay=weight_decay,
                                                                     weight_decay_bias=weight_decay_bias,
                                                                     bias_scale=bias_scale,
                                                                     accum_grad=accum_grad,
                                                                     ema=ema,
                                                                     ema_freq=ema_freq,
                                                                     ema_momentum=ema_momentum,
                                                                     grad_clip=grad_clip)
            if liveplot is not None:
                liveplot.record(epoch, 'lr', [i['lr'] for i in self.optimizer[0].param_groups][-1])
                liveplot.record(epoch, 'train_loss', loss_record)
                liveplot.record(epoch, 'norm', norm_record)
            self.history.add({'norm': norm_record}, prefix = 'train')
            self.history.add({'loss': loss_record}, prefix = 'train')
            self.history.add(metrics_record, prefix = 'train')
            try:
                liveplot.record(epoch, 'train_acc', self.history.records['train_acc'][-1])
            except:
                pass
                
            # ----------------------------------------------
            # Validation stage
            # ----------------------------------------------
            if self.test_loader_length is not None:
                if liveplot is not None:
                    liveplot.redis.set('stage', 'validate')
                    liveplot.val_timer.clear()
                # Validation function
                loss_record, metrics_record = self.validate_fn(liveplot=liveplot,
                                                               batch_size=batch_size,
                                                               epoch=epoch)
                if liveplot is not None:
                    liveplot.record(epoch, 'val_loss', loss_record)
                self.history.add({'loss': loss_record}, prefix = 'val')
                self.history.add(metrics_record, prefix = 'val')
                try:
                    liveplot.record(epoch, 'val_acc', self.history.records['val_acc'][-1])
                except:
                    pass
                        
            # ----------------------------------------------
            # Check point
            # ----------------------------------------------
            
            epoch_str = str(self.epoch)
            if self.val_model.checkpoint_node is not None:
                critic = metrics_record[self.val_model.checkpoint_node.replace('*', '')]
            else:
                critic = None
            
            self.last_state = critic

            if save_checkpoint:
                checkpoint_model_name = os.path.join(
                    self.checkpoint_path, 'checkpoint_' + _id + '_epoch_' + str(self.epoch)
                )
                self.save(checkpoint_model_name)
                
            if self.val_model.checkpoint_state is None:
                self.val_model.checkpoint_state = critic
            
            if critic is not None and self.val_model.checkpoint_mode(critic, self.val_model.checkpoint_state) == critic:
                self.val_model.checkpoint_state = critic
                if save_checkpoint:
                    best_checkpoint_model_name = os.path.join(self.checkpoint_path, 'best_' + _id)
                    self.save(best_checkpoint_model_name)
                epoch_str += '*'
            
            # ----------------------------------------------
            # Post-config
            # ----------------------------------------------
            if liveplot is not None:
                liveplot.plot(self.epoch, self.history, epoch_str)
                liveplot.cali_desc(self.val_model.checkpoint_state)
            
            
            # Steps anneling
            if self.epoch in self.multisteps:
                srange = Annealer.get_srange('lr')
                srange = [i * step_decay for i in srange]
                Annealer.update_attr('lr', 'srange', srange)
                
            # Global epoch annealing
            Annealer._epoch_step()
            
            # Finish this epoch
            self.epoch += 1
            
            # Update logfile
            self.history.update_logfile()
            
        
    except:
        raise

    finally:
        if clear_data:
            self.model.apply(_clear_source)
            self.val_model.apply(_clear_source)
        self.model.apply(_clear_output)
        self.val_model.apply(_clear_output)
        self.unload_gpu()
        _STOP_GPU_MONITOR_ = True
        if liveplot is not None:
            liveplot.redis.set('stage', 'stop')