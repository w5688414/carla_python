import h5py
import cv2

f = h5py.File('../../data8_8/episode_001/data_000000.h5','r')

for num in range(200):
    img=f['rgb'] [num,:,:,:]
    # img = np.split()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('good',img)
    cv2.waitKey(0)
