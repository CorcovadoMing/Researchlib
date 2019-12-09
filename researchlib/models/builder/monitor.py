def MonitorMax(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}, '__MONITOR_MAX__'
    else:
        return {name: (op, inputs)}, '__MONITOR_MAX__'


def MonitorMin(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}, '__MONITOR_MIN__'
    else:
        return {name: (op, inputs)}, '__MONITOR_MIN__'
    
    
def Monitor(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}, '__MONITOR__'
    else:
        return {name: (op, inputs)}, '__MONITOR__'