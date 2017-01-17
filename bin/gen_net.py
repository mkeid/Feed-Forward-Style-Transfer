#!/usr/bin/python
# Mohamed K. Eid (mohamedkeid@gmail.com)

import tensorflow as tf

# Model hyperparameters
decay = .9
epsilon = 1e-3
pad_conv = 'SAME'
pad_residual = 'VALID'

# Generator net architecture reference
layers = [
    {'kind': 'reflection',  'padding_size': 40},
    {'kind': 'conv',        'filters_shape': [9, 9, 3, 32],     'stride': 1},
    {'kind': 'conv',        'filters_shape': [3, 3, 32, 64],    'stride': 2},
    {'kind': 'conv',        'filters_shape': [3, 3, 64, 128],   'stride': 2},

    {'kind': 'residual',    'filters_shape': [3, 3, 128, 128],  'stride': 1},
    {'kind': 'residual',    'filters_shape': [3, 3, 128, 128],  'stride': 1},
    {'kind': 'residual',    'filters_shape': [3, 3, 128, 128],  'stride': 1},
    {'kind': 'residual',    'filters_shape': [3, 3, 128, 128],  'stride': 1},
    {'kind': 'residual',    'filters_shape': [3, 3, 128, 128],  'stride': 1},

    {'kind': 'deconv',      'filters_shape': [3, 3, 64, 128],   'stride': 2},
    {'kind': 'deconv',      'filters_shape': [3, 3, 32, 64],    'stride': 2},

    {'kind': 'output',      'filters_shape': [9, 9, 32, 3],     'stride': 1},
]


class GenNet:
    def __init__(self, weights_path=None):
        self.layers = layers

        if weights_path is not None:
            self.load_net(weights_path)
            self.training = False
        else:
            self.training = True

    @staticmethod
    def get_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0., stddev=.1))

    @staticmethod
    def pad(inputs, size):
        return tf.pad(inputs, [[0, 0], [size, size], [size, size], [0, 0]], "REFLECT")

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    def batch_normalize(self, inputs, num_maps):
        with tf.variable_scope("batch_normalization"):
            # Trainable variables for scaling and offsetting our inputs
            scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32), name='gamma')
            offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32), name='beta')

            # Mean and variances related to our current batch
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

            # Create an optimizer to maintain a 'moving average'
            ema = tf.train.ExponentialMovingAverage(decay=decay, name='ema')

            def ema_retrieve():
                return ema.average(batch_mean), ema.average(batch_var)

            # If the net is being trained, update the average every training step
            def ema_update():
                ema_apply = ema.apply([batch_mean, batch_var])

                # Make sure to compute the new means and variances prior to returning their values
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # Retrieve the means and variances and apply the BN transformation
            mean, var = tf.cond(tf.equal(self.training, True), ema_update, ema_retrieve)
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon)

        return bn_inputs

    # Convolve inputs and return their batch normalized tensor
    def conv_block(self, inputs, maps_shape, padding, stride, pad=False):
        filters = self.get_weights(maps_shape)

        # Reflection pad the input so the dimensions of the final generated image is consistent
        if pad:
            inputs = tf.pad(inputs, [[0, 0], [40, 40], [40, 40], [0, 0]], "CONSTANT")

        filter_maps = tf.nn.conv2d(inputs, filters, [1, stride, stride, 1], padding=padding)
        bn_filter_maps = self.instance_normalize(filter_maps)
        return bn_filter_maps

    #
    def deconv_block(self, inputs, layer):
        filters = self.get_weights(layer['filters_shape'])

        #
        inputs_shape = inputs.get_shape().as_list()
        inputs_batch_size = inputs_shape[0]
        inputs_height = inputs_shape[1]
        inputs_width = inputs_shape[2]

        #
        dim_height = inputs_height * layer['stride']
        dim_width = inputs_width * layer['stride']
        num_output_filters = layer['filters_shape'][2]
        out_shape = tf.pack([inputs_batch_size, dim_height, dim_width, num_output_filters])
        stride = [1, layer['stride'], layer['stride'], 1]

        #
        activations = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, padding=pad_conv, strides=stride)
        bn_activations = self.instance_normalize(activations)

        return tf.nn.relu(bn_activations)

    # Run forward propagation and return the computed image
    def forward_pass(self, img):
        prev_out = img

        for layer in self.layers:
            if layer['kind'] == 'conv' or layer['kind'] == 'output':
                conv = self.conv_block(prev_out, layer['filters_shape'], padding=pad_conv, stride=layer['stride'])
                if layer['kind'] == 'output':
                    prev_out = tf.nn.sigmoid(conv)
                else:
                    prev_out = tf.nn.relu(conv)
            elif layer['kind'] == 'reflection':
                prev_out = GenNet.pad(prev_out, layer['padding_size'])
            elif layer['kind'] == 'residual':
                prev_out = self.residual_block(prev_out, layer)
            elif layer['kind'] == 'deconv':
                prev_out = self.deconv_block(prev_out, layer)

        return prev_out

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    def instance_normalize(self, input):
        # Mean and variances related to our instance
        mean, var = tf.nn.moments(input, [1, 2], keep_dims=True)
        normed_inputs = (input - mean) / (tf.sqrt(var) + epsilon)
        return normed_inputs

    # The residual blocks is comprised of two convolutional layers and aims to add long short-term memory to the network
    def residual_block(self, inputs, layer):
        conv1_out = self.conv_block(inputs, layer['filters_shape'], padding=pad_residual, stride=layer['stride'])
        activations = tf.nn.relu(conv1_out)

        # Retrieve shapes to construct the next set of filters
        activations_shape = activations.get_shape().as_list()
        num_input_maps = activations_shape[3]
        num_output_maps = layer['filters_shape'][3]
        maps_shape = layer['filters_shape'][:2] + [num_input_maps, num_output_maps]

        # Compute the second convolution and make sure that the output shape is appropriate
        conv2_out = self.conv_block(activations, maps_shape, padding=pad_residual, stride=layer['stride'])
        conv2_shape = conv2_out.get_shape().as_list()

        patch_height, patch_width, num_filters = conv2_shape[1:]
        out_shape = [1, patch_height, patch_width, num_filters]

        cropped_inputs = tf.slice(inputs, [0, 1, 1, 0], out_shape)
        return conv2_out + cropped_inputs
