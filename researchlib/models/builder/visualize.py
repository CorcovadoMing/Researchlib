def Visualize(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}, '__VISUAL__'
    else:
        return {name: (op, inputs)}, '__VISUAL__'
