def Output(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}, '__OUTPUT__'
    else:
        return {name: (op, inputs)}, '__OUTPUT__'
