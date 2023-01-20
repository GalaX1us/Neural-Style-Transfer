from utils import *
import time
from model import *

content_image = load_img(CONTENT_PATH)
style_image = load_img(STYLE_PATH)

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', pooling='max')


extractor = StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs, style_targets, content_targets)
    loss += TOTAL_VARIATION_WEIGHT*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,beta_1 = BETA_1,beta_2 = BETA_2, epsilon=EPSILON)
image = tf.Variable(content_image)

start = time.time()

step = 0
for n in range(EPOCHS):
  for m in range(STEPS_PER_EPOCHS):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end-start))
tensor_to_image(image).save(getOutputFileName())



