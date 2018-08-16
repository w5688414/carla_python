import h5py
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
from tqdm import trange
import pickle

def official():
    origin_dir = '/media/kadn/DATA2/AgentHuman/SeqTrain'
    h5files = glob.glob(os.path.join(origin_dir,'*.h5'))
    h5files = h5files[:]
    steer=[[],[],[],[]]

    count0 = 0

    for h5file in tqdm(h5files):
        try:
            data = h5py.File(h5file,'r')
            for i in range(data['targets'].shape[0]):
                target = data['targets'][i]
                if target[24] == 0:
                    count0 += 1
                    print('count %d' % count0, h5file)
                    continue
                steer[np.uint8(target[24]-2)].append(target[0])
            data.close()
        except Exception as exp:
            print(exp)
            print('filename: {}'.format(h5file))

    with open('/home/kadn/official.pk', 'wb') as f:
        pickle.dump(steer, f)


def officialxy():
    origin_dir = '/media/kadn/DATA2/AgentHuman/SeqTrain'
    h5files = glob.glob(os.path.join(origin_dir,'*.h5'))
    h5files = h5files[:]
    x,y = [],[]

    for h5file in tqdm(h5files):
        try:
            data = h5py.File(h5file,'r')
            for i in range(data['targets'].shape[0]):
                target = data['targets'][i]
                if target[24] == 0:
                    continue
                x.append(target[8]/100.0), y.append(target[9]/100.0)
            data.close()
        except Exception as exp:
            print(exp)
            print('filename: {}'.format(h5file))

    with open('/home/kadn/officialxy.pk', 'wb') as f:
        pickle.dump([x,y], f)

def ourdata_steer():
    data_dir = '/home/kadn/AUTODRIVING/data/*/epi*'

    h5files = glob.glob(os.path.join(data_dir,'*.h5'))
    h5files = h5files[:]
    steer = [[], [], [], []]

    for h5file in tqdm(h5files):
        data = h5py.File(h5file,'r')
        for i in range(data['targets'].shape[0]):
            target = data['targets'][i]
            if target[24] == 0:
                continue
            steer[np.uint8(target[24] - 2)].append(target[0])
    with open('/home/kadn/data8_14.pk', 'wb') as f:
        pickle.dump(steer, f)

ourdata_steer()