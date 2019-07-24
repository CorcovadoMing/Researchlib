import numpy as np


def count(y, classes=10):
    hist = [0 for _ in range(classes + 1)]
    for i in y:
        for j in set(i.reshape(100)):
            hist[int(j)] += 1
    print(hist)


def shuffle(l):
    if type(l) == list or type(l) == tuple:
        p = np.random.permutation(len(l[0]))
        return [i[p] for i in l]
    else:
        p = np.random.permutation(len(l))
        return l[p]


def exclude_classes(x, y, exclude):
    if type(exclude) != type([]):
        exclude = [exclude]
    org_x, org_y, new_x, new_y = [], [], [], []
    for i in range(len(x)):
        is_exclude = False
        for ele in exclude:
            if int(ele) in set(y[i].reshape(100)):
                is_exclude = True
                break
        if is_exclude:
            new_x.append(x[i])
            new_y.append(y[i])
        else:
            org_x.append(x[i])
            org_y.append(y[i])
    return org_x, org_y, new_x, new_y


def contains_single(x, y):
    new_x, new_y = [], []
    for i in range(len(x)):
        if len(set(y[i].reshape(100))) == 2:
            new_x.append(x[i])
            new_y.append(y[i])
    return new_x, new_y


def constraint_samples(x, y, num, c):
    hist = [0 for _ in range(11)]
    new_x, new_y = [], []
    for i in range(len(x)):
        if hist[int(c)] >= num:
            break
        for j in set(y[i].reshape(100)):
            hist[int(j)] += 1
        new_x.append(x[i])
        new_y.append(y[i])
    return new_x, new_y


def split_class(x, y, target_class):
    org_x, org_y, tag_x, tag_y = [], [], [], []
    for i in range(len(x)):
        if int(target_class) in set(y[i].reshape(100)):
            tag_x.append(x[i])
            tag_y.append(y[i])
        else:
            org_x.append(x[i])
            org_y.append(y[i])
    return org_x, org_y, tag_x, tag_y


def same_sampling(x, y, num):
    new_x, new_y = [], []
    hist = [0 for _ in range(11)]
    for i in range(len(x)):
        need_add = True
        for j in set(y[i].reshape(100)):
            if hist[int(j)] >= num and int(j) != 0:
                need_add = False
            else:
                need_add = True
                hist[int(j)] += 1
        if need_add:
            new_x.append(x[i])
            new_y.append(y[i])
    return new_x, new_y


def over_sampling(x, y, num):
    ratio = int(num / len(x))
    return x * ratio, y * ratio
