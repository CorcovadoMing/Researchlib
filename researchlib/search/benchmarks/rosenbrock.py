import torch


def Rosenbrock(x):
    result = 0
    for i in range(len(x) - 1):
        result += 100 * ((x[i + 1] + (x[i]**2))**2) + ((x[i] - 1)**2)
    return result
