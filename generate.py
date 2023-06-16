import os
import sys
import numpy as np
from shutil import copy

train = np.load('data-copy/train.npy')
val = np.load('data-copy/val.npy')
test = np.load('data-copy/test.npy')

for i in [train, val, test]:
    for j in i:
        copy(os.path.join('/data/CT_jpg', j), os.path.join('data/imgs', j.split('/')[-1]))
        copy(os.path.join('data-copy/masks', j.split('/')[-1].replace('jpg', 'npy')), os.path.join('data/masks', j.split('/')[-1].replace('jpg', 'npy')))

np.save('data/train.npy', [x.split('/')[-1] for x in train])
np.save('data/val.npy', [x.split('/')[-1] for x in val])
np.save('data/test.npy', [x.split('/')[-1] for x in test])