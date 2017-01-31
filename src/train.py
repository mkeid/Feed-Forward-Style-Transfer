#!/usr/bin/python

"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: trains a generative model for stylizing an unseen image input with a particular style

    Args:
        train: path to image with style to learn
"""

import argparse
import generator
import os
import tensorflow as tf
import trainer

# Model Hyper Params
CONTENT_LAYER = 'conv3_3'
STYLE_LAYERS = {'conv1_2': .25, 'conv2_2': .25, 'conv3_3': .25, 'conv4_3': .25}
EPOCHS = 250000
LEARNING_RATE = .0001
TRAINING_DIMS = {'height': 256, 'width': 256}
PRINT_TRAINING_STATUS = True
PRINT_EVERY_N = 100

# Loss term weights
CONTENT_WEIGHT = 1.
STYLE_WEIGHT = 3.
TV_WEIGHT = .1

# Default image paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = DIR_PATH + '/../lib/images/train2014/'
TRAINED_MODELS_PATH = DIR_PATH + '/../lib/generators/'
TRAIN_PATH = None


# Parse arguments and assign them to their respective global variables
def parse_args():
    global TRAIN_PATH

    # Create flags and assign values to their respective variables
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help="path to image with style to learn")
    args = parser.parse_args()
    TRAIN_PATH = os.path.abspath(args.train)


parse_args()
with tf.Session() as sess:
    with tf.variable_scope('generator'):
        gen = generator.Generator()
        t = trainer.Trainer(sess, gen, TRAIN_PATH, TRAINING_DIMS, PRINT_TRAINING_STATUS, PRINT_EVERY_N)
        t.train(EPOCHS, LEARNING_RATE, CONTENT_LAYER, CONTENT_WEIGHT, STYLE_LAYERS, STYLE_WEIGHT, TV_WEIGHT)
    sess.close()
