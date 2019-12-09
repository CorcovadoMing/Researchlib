def Optimize(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}, '__OPTIMIZE__'
    else:
        return {name: (op, inputs)}, '__OPTIMIZE__'
