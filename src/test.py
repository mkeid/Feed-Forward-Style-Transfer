#!/usr/bin/python

"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: stylizes an image using a generative model trained on a particular style

    Args:
        --input: path to the input image you'd like to apply a style to
        --style: name of style (found in 'lib/generators') to apply to the input
        --out: path to where the stylized image will be created
        --styles: lists trained models available
"""

import argparse
import generator
import helpers
import os
import tensorflow as tf
import time

# Loss term weights
CONTENT_WEIGHT = 1.
STYLE_WEIGHT = 3.
TV_WEIGHT = .1

# Default image paths
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINED_MODELS_PATH = DIR_PATH + '/../lib/generators/'
OUT_PATH = DIR_PATH + '/../output/out_%.0f.jpg' % time.time()
INPUT_PATH, STYLE = None, None


# Parse arguments and assign them to their respective global variables
def parse_args():
    global INPUT_PATH, STYLE, OUT_PATH

    # Create flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="path to the input image you'd like to apply a style to")
    parser.add_argument('--style', help="name of style (found in 'lib/generators') to apply to the input")
    parser.add_argument('--out', default=OUT_PATH, help="path to where the stylized image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    INPUT_PATH = os.path.abspath(args.input)
    STYLE = args.style
    OUT_PATH = args.out


def list_styles():
    styles = ''
    print(styles)

with tf.Session() as sess:
    parse_args()

    # Check if there is a model trained on the given style
    if not os.path.isdir(TRAINED_MODELS_PATH + STYLE):
        print("No trained model with the style '%s' was found." % STYLE)
        list_styles()
        exit(1)

    input_img, _ = helpers.load_img_to(INPUT_PATH, 255, 255)
    input_img = tf.convert_to_tensor(input_img, dtype=tf.float32)
    input_img = tf.expand_dims(input_img, dim=0)

    with tf.variable_scope('generator'):
        gen = generator.Generator()
        gen.build(tf.convert_to_tensor(input_img))
        sess.run(tf.initialize_all_variables())

    ckpt_dir = TRAINED_MODELS_PATH + STYLE
    saved_path = ckpt_dir + "/{}".format(STYLE)
    saver = tf.train.Saver()
    saver.restore(sess, saved_path)

    #
    img = sess.run(gen.output)

    #
    helpers.render(img, path_out=OUT_PATH)
