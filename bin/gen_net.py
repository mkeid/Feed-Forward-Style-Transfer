#!/usr/bin/python
# Mohamed K. Eid (mohamedkeid@gmail.com)

import tensorflow as tf

# Model hyperparameters
decay = .9
epsilon = 1e-8
pad_conv = 'SAME'
pad_resid = 'VALID'


class GenNet:
    def __init__(self, weights_path=None):
        if weights_path is not None:
            self.load_net(weights_path)
            self.training = False
        else:
            self.training = True

    #
    @staticmethod
    def get_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0., stddev=.1), name='weights')

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    @staticmethod
    def instance_normalize(inputs):
        # Mean and variances related to our instance
        mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        normed_inputs = (inputs - mean) / (tf.sqrt(var) + epsilon)
        return normed_inputs

    #
    @staticmethod
    def pad(inputs, size):
        return tf.pad(inputs, [[0, 0], [size, size], [size, size], [0, 0]], "REFLECT")

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    @staticmethod
    def batch_normalize(inputs, num_maps, is_training):
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
            mean, var = tf.cond(tf.equal(is_training, True), ema_update, ema_retrieve)
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon)

        return bn_inputs

    # Convolve inputs and return their batch normalized tensor
    def conv_block(self, inputs, maps_shape, padding, stride, name):
        with tf.variable_scope(name):
            filters = self.get_weights(maps_shape)
            filter_maps = tf.nn.conv2d(inputs, filters, [1, stride, stride, 1], padding=padding)
            num_out_maps = maps_shape[3]
            bias = tf.Variable(tf.constant(0., shape=[num_out_maps]))
            bias_activations = tf.nn.bias_add(filter_maps, bias)

            if name != 'output':
                bias_activations = self.instance_normalize(bias_activations)
                return tf.nn.relu(bias_activations)
            else:
                return tf.nn.sigmoid(bias_activations)

    #
    def deconv_block(self, inputs, maps_shape, stride, name):
        with tf.variable_scope(name):
            filters = self.get_weights(maps_shape)

            #
            inputs_shape = inputs.get_shape().as_list()
            inputs_height = inputs_shape[1]
            inputs_width = inputs_shape[2]

            #
            dim_height = inputs_height * stride
            dim_width = inputs_width * stride
            dim_out = maps_shape[2]
            out_shape = tf.pack([1, dim_height, dim_width, dim_out])
            stride = [1, stride, stride, 1]

            #
            activations = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, padding=pad_conv, strides=stride)
            bias = tf.Variable(tf.constant(0., shape=[dim_out]), name="biases")
            bias_activations = tf.nn.bias_add(activations, bias)
            bn_activations = self.instance_normalize(bias_activations)

            return tf.nn.relu(bn_activations)

    #
    def build(self, img):
        self.padded = self.pad(img, 40)

        self.conv1 = self.conv_block(self.padded, maps_shape=[9, 9, 3, 32], padding=pad_conv, stride=1, name='conv1')
        self.conv2 = self.conv_block(self.conv1, maps_shape=[3, 3, 32, 64], padding=pad_conv, stride=2, name='conv2')
        self.conv3 = self.conv_block(self.conv2, maps_shape=[3, 3, 64, 128], padding=pad_conv, stride=2, name='conv3')

        self.resid1 = self.residual_block(self.conv3, maps_shape=[3, 3, 128, 128], stride=1, name='resid1')
        self.resid2 = self.residual_block(self.resid1, maps_shape=[3, 3, 128, 128], stride=1, name='resid2')
        self.resid3 = self.residual_block(self.resid2, maps_shape=[3, 3, 128, 128], stride=1, name='resid3')
        self.resid4 = self.residual_block(self.resid3, maps_shape=[3, 3, 128, 128], stride=1, name='resid4')
        self.resid5 = self.residual_block(self.resid4, maps_shape=[3, 3, 128, 128], stride=1, name='resid5')

        self.deconv1 = self.deconv_block(self.resid5, maps_shape=[3, 3, 64, 128], stride=2, name='deconv1')
        self.deconv2 = self.deconv_block(self.deconv1, maps_shape=[3, 3, 32, 64], stride=2, name='deconv1')

        self.output = self.conv_block(self.deconv2, maps_shape=[9, 9, 32, 3],
                                      padding=pad_conv, stride=1, name='output')

    # The residual blocks is comprised of two convolutional layers and aims to add long short-term memory to the network
    def residual_block(self, inputs, maps_shape, stride, name):
        with tf.variable_scope(name):
            conv1_out = self.conv_block(inputs, maps_shape, padding=pad_resid, stride=stride, name='c1')
            biases1 = tf.Variable(tf.constant(0., shape=[maps_shape[2]]))
            conv1_out = tf.nn.bias_add(conv1_out, biases1)
            activation = tf.nn.relu(conv1_out)

            # Retrieve shapes to construct the next set of filters
            activation_shape = activation.get_shape().as_list()
            num_input_maps = activation_shape[3]
            num_output_maps = maps_shape[3]
            maps_shape = maps_shape[:2] + [num_input_maps, num_output_maps]

            # Compute the second convolution and make sure that the output shape is appropriate
            conv2_out = self.conv_block(activation, maps_shape, padding=pad_resid, stride=stride, name='c2')
            biases2 = tf.Variable(tf.constant(0., shape=[maps_shape[2]]))
            conv2_out = tf.nn.bias_add(conv2_out, biases2)
            conv2_shape = conv2_out.get_shape().as_list()

            patch_height, patch_width, num_filters = conv2_shape[1:]
            out_shape = [1, patch_height, patch_width, num_filters]

            cropped_inputs = tf.slice(inputs, [0, 1, 1, 0], out_shape)
            return self.instance_normalize(conv2_out) + cropped_inputs
