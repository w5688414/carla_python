import h5py

from PIL import Image

f = h5py.File('train06.h5','r')

num = 500
imgs=f['rgb'][:,:,num]
segs=f['seg'][:,:,num]
a = Image.fromarray(imgs)
seg = Image.fromarray(segs)
a.show()
seg.show()