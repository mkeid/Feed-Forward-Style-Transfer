"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: helper class containing various methods with their functions ranging from image retrieval to auxiliary math helpers
"""

import numpy as np
import tensorflow as tf
from functools import reduce
import skimage
import skimage.io
import skimage.transform
from scipy.misc import toimage


# Compute the content loss given a variable image (x) and a content image (c)
def get_content_loss(x, c, layer):
    with tf.name_scope('get_content_loss'):
        # Get the activated VGG feature maps and return the normalized euclidean distance
        variable_representation = getattr(x, layer)
        photo_representation = getattr(c, layer)

        return get_l2_norm_loss(variable_representation - photo_representation)


# Given an activated filter maps of any particular layer, return its respected gram matrix
def convert_to_gram(filter_maps):
    # Get the dimensions of the filter maps to reshape them into two dimenions
    dimension = filter_maps.get_shape().as_list()
    reshaped_maps = tf.reshape(filter_maps, [dimension[1] * dimension[2], dimension[3]])

    # Compute the inner product to get the gram matrix
    if dimension[1] * dimension[2] > dimension[3]:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        return tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)


# Compute style loss given a variable vgg-out op (variable_model) and a style vgg-out op (style_model)
def get_style_loss(variable_model, style_model, layers):
    with tf.name_scope('get_style_loss'):
        style_layer_losses = [get_style_loss_for_layer(variable_model, style_model, l) for l in layers.keys()]
        style_layer_losses = tf.convert_to_tensor(style_layer_losses)
        style_weights = tf.constant(list(layers.values()))
        weighted_layer_losses = tf.multiply(style_weights, style_layer_losses)
        return tf.reduce_sum(weighted_layer_losses)


# Compute style loss for a layer-out op (l) given the variable vgg-out op (x) and the style vgg-out op (s)
def get_style_loss_for_layer(x, s, l):
    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        x_layer_maps = getattr(x, l)
        s_layer_maps = getattr(s, l)
        x_layer_gram = convert_to_gram(x_layer_maps)
        s_layer_gram = convert_to_gram(s_layer_maps)

        # Make sure the feature grams have the same dimensions
        assert_equal_shapes = tf.assert_equal(x_layer_gram.get_shape(), s_layer_gram.get_shape())
        with tf.control_dependencies([assert_equal_shapes]):
            # Compute and return the normalized gram loss using the gram matrices
            shape = x_layer_maps.get_shape().as_list()
            size = reduce(lambda a, b: a * b, shape) ** 2
            gram_loss = get_l2_norm_loss(x_layer_gram - s_layer_gram)
            return gram_loss / size


# Compute the L2-norm divided by squared number of dimensions
def get_l2_norm_loss(diffs):
    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    return sum_of_squared_diffs / size


# Compute total variation regularization loss term given a variable image (x) and its shape
def get_total_variation(x, shape, smoothing=1.5):
    with tf.name_scope('get_total_variation'):
        # Get the dimensions of the variable image
        height = shape[1]
        width = shape[2]
        size = reduce(lambda a, b: a * b, shape) ** 2

        # Disjoin the variable image and evaluate the total variation
        x_cropped = x[:, :height - 1, :width - 1, :]
        left_term = tf.square(x[:, 1:, :width - 1, :] - x_cropped)
        right_term = tf.square(x[:, :height - 1, 1:, :] - x_cropped)
        smoothed_terms = tf.pow(left_term + right_term, smoothing / 2.)

        return tf.reduce_sum(smoothed_terms) / size


# Returns a numpy array of an image specified by its path
def load_img(path):
    # Load image [height, width, depth]
    img = skimage.io.imread(path) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    shape = list(img.shape)

    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
    return resized_img, shape


# Returns a resized numpy array of an image specified by its path
def load_img_to(path, height=None, width=None):
    # Load image
    img = skimage.io.imread(path) / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]

    if len(img.shape) < 3:
        img = np.dstack((img, img, img))

    return skimage.transform.resize(img, (ny, nx)), [ny, nx, 3]


# Renders the generated image given a tensorflow session and a variable image (x)
def render(img, display=False, path_out=None):
    clipped_img = np.clip(img, 0., 1.)

    if display:
        toimage(np.reshape(clipped_img, img.shape[1:])).show()

    if path_out:
        toimage(np.reshape(clipped_img, img.shape[1:])).save(path_out)
