import h5py

from PIL import Image

f = h5py.File('/media/kadn/DATA2/AgentHuman/SeqTrain/data_03663.h5','r')

num = 100
imgs=f['rgb'][num,:,:,2]
a = Image.fromarray(imgs)
a.show()
# a.save('a.jpg')