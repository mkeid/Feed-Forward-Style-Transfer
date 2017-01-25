#!/usr/bin/python
# Mohamed K. Eid (mohamedkeid@gmail.com)
# Description: Feed-forward implementation of the artistic style-transfer algorithm.

import argparse
import gen_net as gn
import os
import styler
import tensorflow as tf

# Model Hyper Params
content_layer = 'conv3_3'
style_layers = {'conv1_1': .1, 'conv2_2': .1, 'conv3_3': .3, 'conv4_1': .3, 'conv5_1': .2}
epochs = 30000
learning_rate = .0001
training_dims = {'height': 256, 'width': 256}
print_training_status = True


# Loss term weights
content_weight = .05
style_weight = .0000075
tv_weight = 0

# Default image paths
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir_path = dir_path + '/../lib/images/train2014/'
input_path = dir_path + '/../lib/images/content/nyc.jpg'
path_out = None
style = 'great-wave'
train_path = None
trained_models_path = dir_path + '/../lib/generator/'


# Parse arguments and assign them to their respective global variables
def parse_args():
    global input_path, style, train_path, path_out

    # Create flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=input_path, help="path to the input image you'd like to apply a style to")
    parser.add_argument('--style', default=style, help="name of style (found in 'lib/generator') to apply to the input")
    parser.add_argument('--train', default=train_path, help="path to image with style to learn")
    parser.add_argument('--out', default=path_out, help="path to where the stylized image will be created")
    args = parser.parse_args()

    # Assign image paths from the arg parsing
    input_path = os.path.abspath(args.input)
    style = args.style
    train_path = os.path.abspath(args.train)
    path_out = args.out


with tf.Session() as sess:
    parse_args()

    with tf.variable_scope('generator'):
        gen_net = gn.GenNet()
        sty = styler.Styler(sess, gen_net)

        # Given an image, trains a generator net to learn its particular style
        if train_path is not None:
            # Trains a new given style and saves the trained variables
            sty.train_height = training_dims['height']
            sty.train_width = training_dims['width']
            sty.train(epochs, learning_rate, content_layer, content_weight, style_layers, style_weight, tv_weight)

        # Stylize a given image wiht a specified style
        elif input_path is not None and style is not None:
            sty.load_style('great-wave-of-kanagawa')
            sty.stylize(input_path)

    sess.close()
