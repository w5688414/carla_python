import h5py
import cv2

f = h5py.File('/media/kadn/DATA2/AgentHuman/SeqTrain/data_03696.h5','r')

for num in range(200):
    img=f['rgb'] [num,:,:,:]
    # img = np.split()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('good',img)
    cv2.waitKey(0)
