#!/usr/bin/python

"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: trains a generative model for stylizing an unseen image input with a particular style

    Args:

"""

import argparse
import generator
import os
import tensorflow as tf
import trainer

# Model Hyper Params
CONTENT_LAYER = 'conv3_3'
STYLE_LAYERS = {'conv1_2': .25, 'conv2_2': .25, 'conv3_3': .25, 'conv4_3': .25}
EPOCHS = 60000
LEARNING_RATE = .001
TRAINING_DIMS = {'height': 400, 'width': 400}
PRINT_TRAINING_STATUS = True


# Loss term weights
CONTENT_WEIGHT = 1.
STYLE_WEIGHT = 3.
TV_WEIGHT = .1

# Default image paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = DIR_PATH + '/../lib/images/train2014/'
INPUT_PATH = DIR_PATH + '/../lib/images/content/nyc.jpg'
TRAIN_PATH = None
TRAINED_MODELS_PATH = DIR_PATH + '/../lib/generators/'


# Parse arguments and assign them to their respective global variables
def parse_args():
    global TRAIN_PATH

    # Create flags
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help="path to image with style to learn")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    TRAIN_PATH = os.path.abspath(args.train)

# Begin session
with tf.Session() as sess:
    parse_args()

    with tf.variable_scope('generator'):
        gen = generator.Generator()
        trainer = trainer.Trainer(sess, gen)

        # Given an image, trains a generator net to learn its particular style
        trainer.paths['style_file'] = TRAIN_PATH
        trainer.train_height = TRAINING_DIMS['height']
        trainer.train_width = TRAINING_DIMS['width']
        trainer.train(EPOCHS, LEARNING_RATE, CONTENT_LAYER, CONTENT_WEIGHT, STYLE_LAYERS, STYLE_WEIGHT, TV_WEIGHT)
