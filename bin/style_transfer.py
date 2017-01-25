#!/usr/bin/python
# Mohamed K. Eid (mohamedkeid@gmail.com)
# Description: Feed-forward implementation of the artistic style-transfer algorithm.

import argparse
import gen_net as gn
import os
import styler
import tensorflow as tf

# Model Hyper Params
CONTENT_LAYER = 'conv3_3'
STYLE_LAYERS = {'conv1_2': .25, 'conv2_2': .25, 'conv3_3': .25, 'conv4_3': .25}
EPOCHS = 160000
LEARNING_RATE = 1e-4
TRAINING_DIMS = {'height': 400, 'width': 400}
PRINT_TRAINING_STATUS = True


# Loss term weights
CONTENT_WEIGHT = 1.
STYLE_WEIGHT = .03
TV_WEIGHT = 0

# Default image paths
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir_path = dir_path + '/../lib/images/train2014/'
INPUT_PATH = dir_path + '/../lib/images/content/nyc.jpg'
PATH_OUT = None
STYLE = 'great-wave'
TRAIN_PATH = None
trained_models_path = dir_path + '/../lib/generator/'


# Parse arguments and assign them to their respective global variables
def parse_args():
    global INPUT_PATH, STYLE, TRAIN_PATH, PATH_OUT

    # Create flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=INPUT_PATH, help="path to the input image you'd like to apply a style to")
    parser.add_argument('--style', default=STYLE, help="name of style (found in 'lib/generator') to apply to the input")
    parser.add_argument('--train', default=TRAIN_PATH, help="path to image with style to learn")
    parser.add_argument('--out', default=PATH_OUT, help="path to where the stylized image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    INPUT_PATH = os.path.abspath(args.input)
    STYLE = args.style
    TRAIN_PATH = os.path.abspath(args.train)
    PATH_OUT = args.out


with tf.Session() as sess:
    parse_args()

    with tf.variable_scope('generator'):
        gen_net = gn.GenNet()
        sty = styler.Styler(sess, gen_net)

        # Given an image, trains a generator net to learn its particular style
        if TRAIN_PATH is not None:
            # Trains a new given style and saves the trained variables
            sty.paths['style_file'] = TRAIN_PATH
            sty.train_height = TRAINING_DIMS['height']
            sty.train_width = TRAINING_DIMS['width']
            sty.train(EPOCHS, LEARNING_RATE, CONTENT_LAYER, CONTENT_WEIGHT, STYLE_LAYERS, STYLE_WEIGHT, TV_WEIGHT)

        # Stylize a given image wiht a specified style
        elif INPUT_PATH is not None and STYLE is not None:
            sty.load_style('great-wave-of-kanagawa')
            sty.stylize(INPUT_PATH)
