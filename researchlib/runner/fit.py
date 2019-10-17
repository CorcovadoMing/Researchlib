import os
from ..utils import _register_method, plot_montage, _is_port_in_use, inifinity_loop, Annealer, ParameterManager, update_optim
import torch
from .liveplot import Liveplot
from .prefetch import BackgroundGenerator
import pickle
from ..frontend.dashboard import _Dashboard
import numpy as np
import copy


__methods__ = []
register_method = _register_method(__methods__)


def _anneal_policy(anneal_type):
    if anneal_type == 'cosine':
        anneal_policy = Annealer.Cosine
    elif anneal_type == 'linear':
        anneal_policy = Annealer.Linear
    else:
        anneal_policy = Annealer.Fixed
    return anneal_policy

def cov(X):
    X = X/np.sqrt(X.size(0) - 1)
    return X.t() @ X

def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w).to(dtype)

def eigens(patches):
    n,c,h,w = patches.shape
    Σ = cov(patches.reshape(n, c*h*w))
    Λ, V = torch.symeig(Σ, eigenvectors=True)
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

@register_method
def preloop(self, 
            iterations=20):
    
    running_e1, running_e2 = 0, 0
    
    batch_size = 512
    train_loader = self.train_loader.get_generator(batch_size, epochs=3)
    
    for i, (x, y) in enumerate(train_loader):
        if i == iterations:
            break
            
        e1, e2 = eigens(patches(torch.from_numpy(x)))
        running_e1 = 0.9 * running_e1 + 0.1 * e1
        running_e2 = 0.9 * running_e2 + 0.1 * e2
    
    ParameterManager.save_buffer('e1', running_e1)
    ParameterManager.save_buffer('e2', running_e2)
    return self
    

@register_method
def fit(
    self,
    epochs,
    lr = 3e-3,
    policy = 'linear',
    warmup = 5,
    warmup_policy = 'linear',
    flatten = 0,
    flatten_lr = 0,
    metrics = [],
    callbacks = [],
    _id = 'none',
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
        
    
    # ----------------------------------------------
    # Setting loaders
    # ----------------------------------------------
    
    # Few shot learning
    way = parameter_manager.get_param('way', None)
    shot = parameter_manager.get_param('shot', None)
    if way is not None and shot is not None:
        if type(way) == int:
            way = list(range(way))
        support_set = self.train_loader.get_support_set(classes=way, shot=shot)
    else:
        support_set = None
    
    # Load loader
    fp16 = parameter_manager.get_param('fp16', False)
    batch_size = parameter_manager.get_param('batch_size', 512, validator = lambda x: x > 0 and type(x) == int)
    buffered_epochs = epochs + 100
    train_loader = self.train_loader.get_generator(batch_size, epochs=buffered_epochs)
    self.train_loader_length = len(train_loader)
    train_loader = BackgroundGenerator(inifinity_loop(train_loader), fp16=fp16)
    if self.test_loader:
        test_loader = self.test_loader.get_generator(batch_size, epochs=buffered_epochs)
        self.test_loader_length = len(test_loader)
        test_loader = BackgroundGenerator(inifinity_loop(test_loader), fp16=fp16)
    
    liveplot = Liveplot(self.train_loader_length, plot)
        
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
    weight_decay = parameter_manager.get_param('weight_decay', 5e-4)
    if weight_decay > 0:
        weight_decay_policy = parameter_manager.get_param('weight_decay_policy', 'fixed')
        Annealer.set_trace('weight_decay', epochs * iterations, [0, weight_decay], 'iteration', _anneal_policy(weight_decay_policy))
        Annealer._iteration_step(key = 'weight_decay')
        
    # Warmup
    warmup = max(0, warmup)
    if warmup > 0:
        Annealer.set_trace('lr', warmup * iterations, [0, lr], 'iteration', _anneal_policy(warmup_policy))
        Annealer._iteration_step(key = 'lr')
    
    # FP16
    def _to_half(m):
        if isinstance(m, torch.nn.Module) and not isinstance(m, torch.nn.BatchNorm2d):
            m.half()
    if fp16:
        self.model.apply(_to_half)
    
    # For convergence
    bias_scale = parameter_manager.get_param('bias_scale', 1)
    
    # EMA
    ema_freq = max(0, parameter_manager.get_param('ema_freq', 0))
    ema_momentum = parameter_manager.get_param('ema_momentum', 0.99)
    ema = True if ema_freq > 0 else False
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
            if epoch == (warmup + 1):
                Annealer.set_trace('lr', (epochs - warmup - flatten) * iterations, [lr, flatten_lr], 'iteration', _anneal_policy(policy))
                Annealer._iteration_step(key = 'lr')
            
            if epoch == (epochs - flatten + 1):
                Annealer.set_trace('lr', flatten * iterations, [flatten_lr, flatten_lr], 'iteration', _anneal_policy('fixed'))
                Annealer._iteration_step(key = 'lr')
                
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
            # Training function
            loss_record, norm_record = self.train_fn(train_loader, metrics, 
                                                     liveplot=liveplot,
                                                     mmixup_alpha=mmixup_alpha, 
                                                     fixed_mmixup=fixed_mmixup, 
                                                     random_mmixup=random_mmixup,
                                                     epoch=epoch,
                                                     warmup=warmup,
                                                     weight_decay=weight_decay,
                                                     bias_scale=bias_scale,
                                                     ema=ema,
                                                     ema_freq=ema_freq,
                                                     ema_momentum=ema_momentum,
                                                     support_set=support_set,
                                                     way=way,
                                                     shot=shot)
            liveplot.record(epoch, 'lr', [i['lr'] for i in self.optimizer[0].param_groups][-1])
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
            if self.test_loader:
                liveplot.redis.set('stage', 'validate')
                # Validation function
                loss_record = self.validate_fn(test_loader, metrics,
                                               support_set=support_set,
                                               way=way,
                                               shot=shot)
                liveplot.record(epoch, 'val_loss', loss_record)
                self.history_.add({'loss': loss_record}, prefix = 'val')
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
                srange = Annealer.get_srange('lr')
                srange = [i * 0.1 for i in srange]
                Annealer.update_attr('lr', 'srange', srange)
                
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