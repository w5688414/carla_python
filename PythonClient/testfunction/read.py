import h5py
import numpy as np
from PIL import Image
import cv2

f = h5py.File('/media/kadn/DATA2/AgentHuman/SeqTrain/data_03663.h5','r')

num = 100
for num in range(100):
    img=f['rgb'][num,:,:,:]
    # img = np.split()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('good',img)
    cv2.waitKey(0)
# a.save('a.jpg')