#Training Parameters
TOLERANCE = 1e-5
EPOCHS = 500

#Weights
LOSS_TYPE = 1
STYLE_WEIGHTS = [1e0]
CONTENT_WEIGHT = 0.05
TOTAL_VARIATION_WEIGHT = 8.5e-5

#Image
MAX_IMG_SIZE = 512
CONTENT_PATH = 'tubingen.jpg'
STYLE_PATHS = ['starry-night.jpg']
OUTPUT_PATH = './'

#Layers
STYLE_LAYERS = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4',
                    'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4']

CONTENT_LAYERS = ['block5_conv2']