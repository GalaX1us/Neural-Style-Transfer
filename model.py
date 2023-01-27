from params import *
from utils import *

import tensorflow as tf

def gram_matrix(x):
    """
    the gram matrix of an image tensor (feature-wise outer product) using shifted activations

    Args:
        x (np array): image to apply gram matrix on

    Returns:
        np array: transformed image
    """
    gram = tf.linalg.einsum('bijc,bijd->bcd', x - 1, x - 1)
    return gram

class StyleContentModel(tf.keras.Model):
    """
    Allows to have the output of each layer for a specific  input
    """

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        transferL = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        transferL.trainable = False
        
        outputs_dict = dict([(layer.name, layer.output) for layer in transferL.layers])

        style_activations = [outputs_dict[layer_name] for layer_name in style_layers]
        content_activations = [outputs_dict[layer_name] for layer_name in content_layers]

        activations = style_activations + content_activations

        self.vgg = tf.keras.Model(transferL.input, activations)

        self.style_layer_names = style_layers
        self.content_layer_names = content_layers

        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)

    def call(self, inputs):

        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layer_names, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layer_names, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
    
def style_loss(style, combination, size):
    """
    Compute the style loss

    Args:
        style (np array): target image
        combination (np array): trained image
        size (tuple): size of the image

    Returns:
        np array: loss value
    """
    channels = 3
    return tf.reduce_sum(tf.square(style - combination)) / (4. * (channels ** 2) * (size ** 2))

def content_loss(base, combination, size):
    """
    Compute content loss

    Args:
        base (np array): target image
        combination (np array): trained image
        size (tuple): size of the image

    Returns:
        np array: loss value
    """
    channels = 3
    if LOSS_TYPE == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif LOSS_TYPE == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    return multiplier * tf.reduce_sum(tf.square(combination - base))

# 
def total_variation_loss(x):
    """
    The total variation loss is designed to keep the generated image locally coherent
    by reducing high frequency artifacts

    Args:
        x (np array): input image

    Returns:
        np array: loss value
    """
    a = tf.square(
        x[:, :-1, :-1, :] - x[:, 1:, :-1, :]
    )
    b = tf.square(
        x[:, :-1, :-1, :] - x[:, :-1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))




def compute_loss(input, outputs, content_target, style_targets):
    """
    Compute the overall loss of the image

    Args:
        input (np array): trained image
        outputs (dict): outputs of each layer of the model with this input
        content_target (np array): self explanatory
        style_targets (np array): self explanatory

    Returns:
        np array: overall loss value
    """
    style_combined_outputs = outputs['style']
    content_combined_outputs = outputs['content']
    h,w,c = input.shape[1:]
    size = h*w

    # Content losses
    content_losses = CONTENT_WEIGHT * tf.add_n([content_loss(content_target[name], content_combined_outputs[name], size)
                                                            for name in content_combined_outputs.keys()])

    num_style_layers = len(STYLE_LAYERS)
    num_style_references = len(style_targets)

    # Style losses (Cross layer loss)
    style_losses = []
    for style_img_id in range(num_style_references):
        style_features = style_targets[style_img_id]

        sl_i = 0.
        for feature_layer_id in range(num_style_layers - 1):
            target_feature_layer = style_features[STYLE_LAYERS[feature_layer_id]]
            style_output = style_combined_outputs[STYLE_LAYERS[feature_layer_id]]

            sl1 = style_loss(target_feature_layer, style_output, size)

            target_feature_layer = style_features[STYLE_LAYERS[feature_layer_id + 1]]
            style_output = style_combined_outputs[STYLE_LAYERS[feature_layer_id + 1]]

            sl2 = style_loss(target_feature_layer, style_output, size)

            # Geometric loss scaling
            sl_i = sl_i + (sl1 - sl2) * (STYLE_WEIGHTS[style_img_id] / (2 ** (num_style_layers - 1 - (feature_layer_id + 1))))

        style_losses.append(sl_i)

    style_losses = tf.add_n(style_losses)

    # Total Variation Losses
    tv_losses = TOTAL_VARIATION_WEIGHT * total_variation_loss(input)

    return content_losses, style_losses, tv_losses

def get_feature_representations(model, content_path, style_paths):
  """Helper function to compute our content and style feature representations.

  This function will simply load and preprocess both the content and style 
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image
    
  Returns:
    returns the style features and the content features. 
  """
  
  content_image = load_and_process_img(content_path)
  content_features = model(content_image)['content']
  
  style_features = []
  for path in style_paths: 
    style_image = load_and_process_img(path)
    style_features.append(model(style_image)['style'])

  return style_features, content_features