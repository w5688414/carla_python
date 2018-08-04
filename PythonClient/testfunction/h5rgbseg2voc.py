import numpy as np
import h5py
import os
from matplotlib.image import imsave
from PIL import Image

def mkdir(path):
    path = path.strip()
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return True

dir = '/home/kadn/dataTrain'
rgb_output_dir = '/home/kadn/VOCdevkit/Images'
seg_output_dir = '/home/kadn/VOCdevkit/Labels'
mkdir(rgb_output_dir)
mkdir(seg_output_dir)
numofrecords = 49
numofimgs = 0

episodes = [os.path.join(dir,name) for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
# print(episode)
for episode in episodes:
    files = os.listdir(episode)
    for file in files:
        f = h5py.File(os.path.join(episode, file), 'r')
        for i in range(numofrecords):
            img = f['CameraRGB'][:, :, 4*i]
            imgname = 'rgb_{:0>6}.jpg'.format(numofimgs)
            imgdir = os.path.join(rgb_output_dir, imgname)
            imgI = Image.fromarray(img)
            imgI.save(imgdir)

            # seg = f['CameraSemSeg'][:,:,4*i]
            # segname = 'rgb_{:0>6}.png'.format(numofimgs)
            # segdir = os.path.join(seg_output_dir,segname)
            # segI = Image.fromarray(seg)
            # segI.save(segdir)
            numofimgs += 1
