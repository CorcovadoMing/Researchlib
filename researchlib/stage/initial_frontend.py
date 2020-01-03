from ..frontend.dashboard import _Dashboard
from ..utils import _is_port_in_use
import redis
import pickle


def _initialize_redis(r, variable, init_value, need_encode = False):
    try:
        result = r.get(variable)
        assert result is not None
        if need_encode:
            result = pickle.loads(result)
    except:
        if need_encode:
            init_value = pickle.dumps(init_value)
        r.set(variable, init_value)

        
def initial_frontend():
    if _is_port_in_use(8050):
        print()
        print('* Visit dashboard at http://<ip>:8050')
    else:
        r = redis.Redis()
        _initialize_redis(r, 'progress', 0)
        _initialize_redis(r, 'desc', '')
        _initialize_redis(r, 'stage', 'stop')
        _initialize_redis(
            r,
            'history', {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            },
            need_encode = True
        )
        _initialize_redis(r, 'experiment', [], need_encode = True)
        del r
        dash = _Dashboard(verbose = False)
        dash.start()
        print()
        print('* Dashboard is open at http://<ip>:8050')
