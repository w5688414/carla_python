import numpy as np
import h5py
import os
from matplotlib.image import imsave
from PIL import Image

img_dir = '/home/kadn/VOCdevkit/Images'

filecount = os.listdir(img_dir)
numofpics = len(filecount)

train =  int(0.6*numofpics)
val = int(0.2*numofpics)
test = int(0.2*numofpics)

def mkdir(path):
    path = path.strip()
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return True
# imgname = 'rgb_{:0>6}.jpg'.format(numofimgs)
# segname = 'seg_{:0>6}.jpg'.format(numofimgs)
mkdir('/home/kadn/VOCdevkit/list')

filepath = '/home/kadn/VOCdevkit/list/train.txt'
with open(filepath,'w') as f:
    for i in range(train):
        imgname = 'rgb_{:0>6}'.format(i)
        f.write(imgname+'\n')


filepath = '/home/kadn/VOCdevkit/list/val.txt'
with open(filepath,'w') as f:
    for i in range(val):
        imgname = 'rgb_{:0>6}'.format(i+train)
        f.write(imgname+'\n')


filepath = '/home/kadn/VOCdevkit/list/trainval.txt'
with open(filepath,'w') as f:
    for i in range(train+val):
        imgname = 'rgb_{:0>6}'.format(i)
        f.write(imgname+'\n')

