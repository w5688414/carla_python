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

def official_steer():
    f=open('/home/kadn/data8_14.pk','rb')
    steer = pickle.load(f)

    now1 = []
    now1.extend(steer[3])

    now2 = []
    now2.extend(steer[1])
    now2.extend(steer[2])

    now3 = []
    now4 = []

    plt.subplot(241)
    plt.hist(steer[0],41)
    plt.title('follow')
    plt.subplot(242)
    plt.hist(steer[1],41)
    plt.title('left')
    plt.subplot(243)
    plt.hist(steer[2],41)
    plt.title('right')
    plt.subplot(244)
    plt.hist(steer[3],41)
    plt.title('straight')
    plt.subplot(245)
    plt.title('follow')
    plt.subplot(246)
    plt.subplot(247)
    plt.subplot(248)
    plt.hist(now2,41)
    plt.title('left plus right')
    plt.show()

def official_xy():
    with open('/home/kadn/officialxy.pk', 'rb') as f:
        data = pickle.load(f)

    x = data[0]
    y = data[1]


    plt.show()

official_steer()