from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from PIL import Image
from utils import *
from params import *
from model import *
import time
from LBFGS import LBFGSOptimizer

extractor = StyleContentModel(style_layers=STYLE_LAYERS, content_layers=CONTENT_LAYERS)

base_image = load_and_process_img(CONTENT_PATH)

style_targets_list, content_target = get_feature_representations(extractor, CONTENT_PATH, STYLE_PATHS)

x = tf.Variable(base_image, trainable=True)

class InputWrapper(tf.keras.Model):
    def __init__(self, x: tf.Variable):
        super(InputWrapper, self).__init__()

        self.x = x

        self.vgg = extractor
        self.vgg.trainable = False

    def call(self, inputs, training=None):
        outputs = self.vgg(self.x)
        return outputs

x_wrapper = InputWrapper(x)

@tf.function  # (tracing will be done by LBFGS for us)
def loss_wrapper(model):
    outputs = model(x)
    content_losses, style_losses, tv_losses = compute_loss(x, outputs, content_target, style_targets_list)
    loss = content_losses + style_losses + tv_losses
    return loss

prev_min_val = -1
start_time = time.time()


def save_image_callback(model, info_dict=None):
    """
    Callback function to get infos while training
    """
    global prev_min_val, start_time

    info_dict = info_dict or {}
    loss_value = info_dict.get('loss', None)
    i = info_dict.get('iter', -1)

    if loss_value is not None:
        loss_val = loss_value.numpy()

        if prev_min_val == -1:
            prev_min_val = loss_val

        improvement = (prev_min_val - loss_val) / prev_min_val * 100

        print("Current loss value:", loss_val, " Improvement : %0.3f" % improvement, "%")
        prev_min_val = loss_val

    last_save = info_dict.get('last_save', False)
    if (i + 1) % 100 == 0 or last_save:
        img = model.x.numpy()
        # save current generated image
        img = deprocess_img(img)

        if not last_save:
            fname = OUTPUT_PATH + "out" + "_at_iteration_%d.png" % (i + 1)
        else:
            fname = OUTPUT_PATH + "out" + "_final.png"

        imsave(fname, img)
        end_time = time.time()
        print("Image saved as", fname)
        print("Iteration %d completed in %ds" % (i + 1, end_time - start_time))


optimizer = LBFGSOptimizer(max_iterations=EPOCHS, tolerance=TOLERANCE)

# Add a save image callback to this
optimizer.register_callback(save_image_callback)

optimizer.minimize(loss_wrapper, x_wrapper)

save_image_callback(x_wrapper, info_dict={'last_save': True})