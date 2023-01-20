import PIL.Image
import tensorflow as tf
import numpy as np
from params import *

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def get_name_from_path(path):
    img = path.split('/')[-1]
    name = img.split('.')[0]
    return name

def getOutputFileName():
    file_name = "{}-X-{}-{}-{}-{}.png".format(
        get_name_from_path(CONTENT_PATH), 
        get_name_from_path(STYLE_PATH),
        STYLE_WEIGHT, CONTENT_WEIGHT, TOTAL_VARIATION_WEIGHT)
    return OUTPUT_PATH + file_name

def load_img(path_to_img):
  max_dim = 1200
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3, dtype=tf.float32)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img