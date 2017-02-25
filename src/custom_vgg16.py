"""
    Author: Chris (https://github.com/machrisaa), modified by Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: tensorflow implemention of VGG 16 and VGG 19 based on tensorflow-vgg16
"""

import os
import tensorflow as tf
import numpy as np
import inspect
import urllib.request

VGG_MEAN = [103.939, 116.779, 123.68]
data = None
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_name = dir_path + "/../lib/descriptor/vgg16.npy"
weights_url = "https://www.dropbox.com/s/gjtfdngpziph36c/vgg16.npy?dl=1"


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        global data

        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, weights_name)

            if os.path.exists(path):
                vgg16_npy_path = path
            else:
                print("VGG16 weights were not found in the project directory")

                answer = 0
                while answer is not 'y' and answer is not 'N':
                    answer = input("Would you like to download the 528 MB file? [y/N] ").replace(" ", "")

                # Download weights if yes, else exit the program
                if answer == 'y':
                    print("Downloading. Please be patient...")
                    urllib.request.urlretrieve(weights_url, weights_name)
                    vgg16_npy_path = path
                elif answer == 'N':
                    print("Exiting the program..")
                    exit(0)

        if data is None:
            data = np.load(vgg16_npy_path, encoding='latin1')
            self.data_dict = data.item()
            print("VGG net weights loaded")

        else:
            self.data_dict = data.item()

    def build(self, rgb, shape):
        rgb_scaled = rgb * 255.0
        num_channels = shape[2]
        channel_shape = shape
        channel_shape[2] = 1

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        assert red.get_shape().as_list()[1:] == channel_shape
        assert green.get_shape().as_list()[1:] == channel_shape
        assert blue.get_shape().as_list()[1:] == channel_shape

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        shape[2] = num_channels
        assert bgr.get_shape().as_list()[1:] == shape

        self.conv1_1 = self.__conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.__conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.__avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.__conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.__conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.__avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.__conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.__conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.__conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.__avg_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.__conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.__conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.__conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.__avg_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.__conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.__conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.__conv_layer(self.conv5_2, "conv5_3")

        self.data_dict = None

    def __avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.__get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.__get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def __get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def __get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def __get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")