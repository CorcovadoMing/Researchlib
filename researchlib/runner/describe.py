import copy
from ..utils import _register_method, ParameterManager

__methods__ = []
register_method = _register_method(__methods__)

@register_method
def describe(self):
    def _describe_model(model_dict):
        query = {}
        keys = [
            'do_norm', 'pool_freq', 'preact', 'filter_policy', 'filters', 'type',
            'total_blocks', 'op', 'unit'
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
        keys = ['self_iterative', 'mmixup_alpha', 'policy', 'lr', 'epochs']
        for key, value in fit_dict.items():
            if key in keys:
                query[key] = copy.deepcopy(value)
        return query

    def _get_best_metrics(runner, metrics):
        index = runner.history_.records['saved'].index('*', -1)
        return runner.histroy_.records['val_' + str(metrics)][index]

    keys = [
        'swa', 'swa_start', 'larc', 'fp16', 'augmentation_list', 'weight_decay',
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

    query['train_loader'] = str(query['train_loader'].__class__.__name__) + ': ' + str(query['train_loader'].name)

    query['optimizer'] = self.__class__.__runner_settings__['optimizer']
    query['monitor_state'] = self.__class__.__runner_settings__['monitor_state']

    query['num_params'] = self.num_params

    try:
        query['best_state'] = self.__dict__['monitor']
    except:
        query['best_state'] = _get_best_metrics(self, query['monitor_state'])

    query['model'] = {}
    for i, j in self.__class__.__model_settings__.items():
        query['model'].setdefault(i, {})
        query['model'][i] = _describe_model(j)
    query['fit'] = {}
    for i, j in self.__class__.__fit_settings__.items():
        query['fit'].setdefault(i, {})
        query['fit'][i] = _describe_fit(j)
    query.update(ParameterManager.params)
    return query