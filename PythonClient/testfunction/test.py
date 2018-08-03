import numpy as np
import h5py

f = h5py.File("train.h5", "w")
rgb_file = f.create_dataset("rgb", (200, 88, 200), np.uint8)
targets_file = f.create_dataset("targets", (200, 28), np.float32)
index_file = 0

for i in range(200):
    rgb_data = np.zeros((200,88))
    targets_data = np.ones((1,28))
    rgb_file[:,:,index_file] = rgb_data
    targets_file[index_file,:] = targets_data
    index_file = index_file + 1

