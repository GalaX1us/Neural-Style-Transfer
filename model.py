import tensorflow as tf
from params import *

content_layers = ['block4_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    
    outputs = self.vgg(preprocessed_input) #tt les infos
    
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]
    
    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets):
  
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= STYLE_WEIGHT / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= CONTENT_WEIGHT / num_content_layers
    loss = style_loss + content_loss
    return loss

# def style_content_loss(outputs):
  
#     style_outputs = outputs['style']
#     content_outputs = outputs['content']
    
#     style_loss = 0
#     content_loss = 0
    
#     for name in content_outputs.keys():
#       T = content_targets[name]
#       O = content_outputs[name]
#       _, h, w, d = O.shape
      
#       N = h*w
#       M = d
    
#       content_loss += (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((O - T), 2))
    
#     content_loss *= content_weight / num_content_layers

#     for name in style_outputs.keys():
#       T = style_targets[name]
#       O = style_outputs[name]
      
#       _, h, w, d = O.shape
      
#       N = h*w
#       M = d
      
#       G = gram_matrix(O)

#       style_loss += (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-T), 2))
    
#     style_loss *= style_weight / num_style_layers
    
#     loss = style_loss + content_loss
    
#     return loss

# for name in style_targets.keys():
#   style_targets[name] = gram_matrix(style_targets[name])