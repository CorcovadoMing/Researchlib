from researchlib.single_import import *
from torch.optim import *

'''
# Load dataset
train_loader = FromPublic('cifar10', 'train', batch_size=512, pin_memory=True)
test_loader = FromPublic('cifar10', 'test', batch_size=512, pin_memory=True)
'''

import numpy as np
x = np.random.random((320, 25, 1, 32, 32))
y = np.random.random((320, 25, 1))
train_loader = FromNumpy(x, y, batch_size=32)


# Network models
model = builder([
    TimeDistributed(nn.Conv2d(1, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
    TimeDistributed(nn.Conv2d(3, 3, 3)),
])
model = model.cuda()
print(model)

for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    data, target = data.cuda(), target.cuda()
    out = model(data)
    print(out.shape)

'''
# Learning
optimizer = Adam(model.parameters(), lr=1e-6)
runner = Runner(model, train_loader, test_loader, optimizer)
#runner.find_lr(plot=True)
runner.fit(1, 5e-4)
'''