"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: helper class containing various methods with their functions ranging from image retrieval to auxiliary math helpers
"""

import logging
import numpy as np
import sys
from functools import reduce

from scipy.misc import toimage
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf


def config_logging():
    """Configure the python logger."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def get_content_loss(variable_img, content_img, layer):
    """Compute the content loss given a variable image (x) and a content image (c).
    
    Args:
        variable_img: 4D tensor representing the variable image aimed to be generated
        content_img: 4D tensor representing the image expected to retrieve the content representation from
        layer: string representing the vgg layer for computing the encoding
        
    Returns:
        normed_loss: tensor representing the l2 normalized content loss value
    """

    with tf.name_scope('get_content_loss'):
        # Get the activated VGG feature maps and return the normalized euclidean distance
        variable_representation = getattr(variable_img, layer)
        photo_representation = getattr(content_img, layer)

        normed_loss = get_l2_norm_loss(variable_representation - photo_representation)
        return normed_loss


def convert_to_gram(feature_maps):
    """Given an activated filter maps of any particular layer, return its respected gram matrix..
    
    Args:
        feature_maps: 4D tensor representing vgg feature maps
        
    Returns:
        gram: tensor representing the proportional correlations across feature maps through the inner product
    """

    feature_shape = feature_maps.get_shape().as_list()
    new_shape = [feature_shape[1] * feature_shape[2], feature_shape[3]]
    reshaped_maps = tf.reshape(feature_maps, new_shape)

    if feature_shape[1] * feature_shape[2] > feature_shape[3]:
        gram = tf.matmul(reshaped_maps, reshaped_maps, transpose_a=True)
    else:
        gram = tf.matmul(reshaped_maps, reshaped_maps, transpose_b=True)
    return gram


def get_style_loss(variable_model, style_model, layers):
    """Compute style loss given a variable vgg-out op (variable_model) and a style vgg-out op (style_model).
    
    Args:
        variable_model: vgg model used for the variable image
        style_model: vgg model used for the style image
        layers: list of vgg layers
        
    Returns:
        loss: float tensor representing the total style loss
    """

    with tf.name_scope('get_style_loss'):
        style_layer_losses = [get_style_loss_for_layer(variable_model, style_model, layer) for layer in layers.keys()]
        style_layer_losses = tf.convert_to_tensor(style_layer_losses)
        style_weights = tf.constant(list(layers.values()))
        weighted_layer_losses = tf.multiply(style_weights, style_layer_losses)
        loss = tf.reduce_sum(weighted_layer_losses)
        return loss


def get_style_loss_for_layer(variable_img, style_img, layer):
    """Compute style loss for a layer-out op (l) given the variable vgg-out op (x) and the style vgg-out op (s).
    
    Args:
        variable_img: 4D tensor representing the variable image vgg encodings
        style_img: 4D tensor representing the style image vgg encodings
        layer: string representing the vgg layer
        
    Returns:
        loss: float tensor representing the style loss for the given layer
    """

    with tf.name_scope('get_style_loss_for_layer'):
        # Compute gram matrices using the activated filter maps of the art and generated images
        x_layer_maps = getattr(variable_img, layer)
        s_layer_maps = getattr(style_img, layer)
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


def get_l2_norm_loss(diffs):
    """Compute the L2-norm divided by squared number of dimensions.
    
    Args:
        diffs: float tensor to normalize
        
    Returns:
        loss: float tensor representing the l2 normalized loss
    """

    shape = diffs.get_shape().as_list()
    size = reduce(lambda x, y: x * y, shape) ** 2
    sum_of_squared_diffs = tf.reduce_sum(tf.square(diffs))
    loss = sum_of_squared_diffs / size
    return loss


def get_total_variation(variable_img, shape, smoothing=1.5):
    """Compute total variation regularization loss term given a variable image (x) and its shape.
    
    Args:
        variable_img: 4D tensor representing the variable image
        shape: list representing the variable image shape
        smoothing: smoothing parameter for penalizing large variations
        
    Returns:
        variation: float tensor representing the total variation for a given image
    """

    with tf.name_scope('get_total_variation'):
        # Get the dimensions of the variable image
        height = shape[1]
        width = shape[2]
        size = reduce(lambda a, b: a * b, shape) ** 2

        # Disjoin the variable image and evaluate the total variation
        x_cropped = variable_img[:, :height - 1, :width - 1, :]
        left_term = tf.square(variable_img[:, 1:, :width - 1, :] - x_cropped)
        right_term = tf.square(variable_img[:, :height - 1, 1:, :] - x_cropped)
        smoothed_terms = tf.pow(left_term + right_term, smoothing / 2.)
        variation = tf.reduce_sum(smoothed_terms) / size
        return variation


def load_img(path):
    """Returns a numpy array of an image specified by its path.
    
    Args:
        path: string representing the file path of the image to load
        
    Returns:
        resized_img: numpy array representing the loaded RGB image
        shape: the image shape
    """

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


def load_img_to(path, height=None, width=None):
    """Returns a resized numpy array of an image specified by its path.
    
    Args:
        path: string representing the file path of the image to load
        height: int representing the height value to scale image
        width: int representing width value to scale image
        
    Returns:
        img: numpy array representing the loaded RGB image
    """

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


def render(img, display=False, path_out=None):
    """Renders the generated image given a tensorflow session and a variable image (x).
    
    Args:
        img: numpy array representing an RGB image to render
        display: boolean value representing whether or not to display the image on screen using toimage
        path_out: string representing the file path to save the image rendering
    """

    clipped_img = np.clip(img, 0., 1.)

    if display:
        toimage(np.reshape(clipped_img, img.shape[1:])).show()

    if path_out:
        toimage(np.reshape(clipped_img, img.shape[1:])).save(path_out)
