import numpy as np
import imageio
from PIL import Image
from params import *

import tensorflow as tf
  
def imresize(img, size, interp='bilinear'):
  """
  Used to resize images to a specific size

  Args:
      img (Image): image to resize
      size (tuple): size
      interp (str, optional): interpolation style. Defaults to 'bilinear'.

  Returns:
      _type_: _description_
  """
  if interp == 'bilinear':
      interpolation = Image.BILINEAR
  elif interp == 'bicubic':
      interpolation = Image.BICUBIC
  else:
      interpolation = Image.NEAREST

  size = (size[1], size[0])

  if type(img) != Image:
      img = Image.fromarray(img, mode='RGB')

  img = np.array(img.resize(size, interpolation))
  return img
    
def imsave(path, img):
  """
  Save an image

  Args:
      path (string): path where the image will be saved
      img (Image): image to be saved
  """
  imageio.imwrite(path, img)
  return

def load_img(path_to_img):
  """
  Load an image from a path, resize it to match the maximum dim and format it
  for training

  Args:
      path_to_img (string): path to the image

  Returns:
      numpy array: image as a np array
  """
  max_dim = MAX_IMG_SIZE
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  
  img *= 255

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  img = img.numpy()
  
  return img

def load_and_process_img(path_to_img):
  """
  Load and preprocess the image so it can be fed into the model

  Args:
      path_to_img (path): path to the image

  Returns:
      np array: preprocessed image
  """
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

def deprocess_img(processed_img):
  """
  Undo the preprocessing on the image

  Args:
      processed_img (np array): image to be deprocessed

  Raises:
      ValueError: Invalid input to deprocessing image

  Returns:
      np array: deprocessed image
  """
  if len(processed_img.shape) == 4:
    processed_img = np.squeeze(processed_img, 0)
  assert len(processed_img.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(processed_img.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessing step
  processed_img[:, :, 0] += 103.939
  processed_img[:, :, 1] += 116.779
  processed_img[:, :, 2] += 123.68
  processed_img = processed_img[:, :, ::-1]

  processed_img = np.clip(processed_img, 0, 255).astype('uint8')
  return processed_img

def clip_image(image):
  return tf.clip_by_value(image, clip_value_min=-128, clip_value_max=128.0)