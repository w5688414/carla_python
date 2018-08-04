import tensorflow as tf
from PIL import Image
from matplotlib.image import imsave,imread

def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
image_filename = '/home/kadn/VOCdevkit/Images/rgb_000000.png'
a= imread(image_filename)


pasca_path = '2007_000032.jpg'
image_data = tf.gfile.FastGFile(pasca_path, 'r').read()