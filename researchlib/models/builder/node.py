def Node(name, op, *inputs):
    if len(inputs) == 0:
        return {name: op}
    else:
        return {name: (op, inputs)}