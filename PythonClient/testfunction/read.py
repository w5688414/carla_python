import h5py

from PIL import Image

f = h5py.File('data_000007.h5','r')

num = 100
imgs=f['CameraRGB'][:,:,num]
a = Image.fromarray(imgs)
a.show()
a.save('a.jpg')