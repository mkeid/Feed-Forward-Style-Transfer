"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: generative model with the architectural specifications suited for artistic style transfer
"""
import tensorflow as tf

# Model hyperparameters
DECAY = .9
EPSILON = 1e-8


class Generator:
    def __init__(self, is_training=True):
        self.training = is_training

    # Constructs the generative network's layers. Normally called after initialization.
    def build(self, img):
        self.padded = self.__pad(img, 40)

        self.conv1 = self.__conv_block(self.padded, maps_shape=[9, 9, 3, 32], stride=1, name='conv1')
        self.conv2 = self.__conv_block(self.conv1, maps_shape=[3, 3, 32, 64], stride=2, name='conv2')
        self.conv3 = self.__conv_block(self.conv2, maps_shape=[3, 3, 64, 128], stride=2, name='conv3')

        self.resid1 = self.__residual_block(self.conv3, maps_shape=[3, 3, 128, 128], stride=1, name='resid1')
        self.resid2 = self.__residual_block(self.resid1, maps_shape=[3, 3, 128, 128], stride=1, name='resid2')
        self.resid3 = self.__residual_block(self.resid2, maps_shape=[3, 3, 128, 128], stride=1, name='resid3')
        self.resid4 = self.__residual_block(self.resid3, maps_shape=[3, 3, 128, 128], stride=1, name='resid4')
        self.resid5 = self.__residual_block(self.resid4, maps_shape=[3, 3, 128, 128], stride=1, name='resid5')

        self.conv4 = self.__upsample_block(self.resid5, maps_shape=[3, 3, 64, 128], stride=2, name='conv4')
        self.conv5 = self.__upsample_block(self.conv4, maps_shape=[3, 3, 32, 64], stride=2, name='conv5')
        self.conv6 = self.__conv_block(self.conv5, maps_shape=[9, 9, 32, 3], stride=1, name='conv6', activation=None)

        self.output = tf.nn.sigmoid(self.conv6)

    # Returns a variable for weights wiht a specified filters shape
    @staticmethod
    def __get_weights(shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0., stddev=.1), dtype=tf.float32)

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    @staticmethod
    def __instance_normalize(inputs):
        with tf.variable_scope('instance_normalization'):
            batch, height, width, channels = [_.value for _ in inputs.get_shape()]
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            shift = tf.Variable(tf.constant(.1, shape=[channels]))
            scale = tf.Variable(tf.ones([channels]))
            normalized = (inputs - mu) / (sigma_sq + EPSILON) ** .5

            return scale * normalized + shift

    # Pads input of the image so the output is the same dimensions even after upsampleolution
    @staticmethod
    def __pad(inputs, size):
        return tf.pad(inputs, [[0, 0], [size, size], [size, size], [0, 0]], "REFLECT")

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    @staticmethod
    def __batch_normalize(inputs, num_maps, is_training):
        with tf.variable_scope("batch_normalization"):
            # Trainable variables for scaling and offsetting our inputs
            scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32))
            offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32))

            # Mean and variances related to our current batch
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

            # Create an optimizer to maintain a 'moving average'
            ema = tf.train.ExponentialMovingAverage(decay=DECAY)

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
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, EPSILON)

        return bn_inputs

    # Convolve inputs and return their batch normalized tensor
    def __conv_block(self, inputs, maps_shape, stride, name, norm=True, padding='SAME', activation=tf.nn.relu):
        with tf.variable_scope(name):
            if name == 'output':
                activation = tf.nn.sigmoid

            filters = self.__get_weights(maps_shape)
            filter_maps = tf.nn.conv2d(inputs, filters, [1, stride, stride, 1], padding=padding)
            num_out_maps = maps_shape[3]
            bias = tf.Variable(tf.constant(.1, shape=[num_out_maps]))
            filter_maps = tf.nn.bias_add(filter_maps, bias)

            if norm:
                filter_maps = self.__instance_normalize(filter_maps)

            if activation:
                return activation(filter_maps)
            else:
                return filter_maps

    # Upsamples inputs using transposed convolution
    def __upsample_block(self, inputs, maps_shape, stride, name):
        with tf.variable_scope(name):
            filters = self.__get_weights(maps_shape)

            # Get dimensions to use for the upsample operator
            batch, height, width, channels = inputs.get_shape().as_list()
            out_height = height * stride
            out_width = width * stride
            out_size = maps_shape[2]
            out_shape = tf.stack([batch, out_height, out_width, out_size])
            stride = [1, stride, stride, 1]

            # Upsample and normalize the biased outputs
            upsample = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, strides=stride)
            bias = tf.Variable(tf.constant(.1, shape=[out_size]))
            upsample = tf.nn.bias_add(upsample, bias)
            bn_maps = self.__instance_normalize(upsample)

            return tf.nn.relu(bn_maps)

    # The residual blocks is comprised of two convolutional layers and aims to add long short-term memory to the network
    def __residual_block(self, inputs, maps_shape, stride, name):
        with tf.variable_scope(name):
            conv1 = self.__conv_block(inputs, maps_shape, stride=stride, padding='VALID', name='c1')
            conv2 = self.__conv_block(conv1, maps_shape, stride=stride, padding='VALID', name='c2', activation=None)

            batch = inputs.get_shape().as_list()[0]
            patch_height, patch_width, num_filters = conv2.get_shape().as_list()[1:]
            out_shape = tf.stack([batch, patch_height, patch_width, num_filters])
            cropped_inputs = tf.slice(inputs, [0, 1, 1, 0], out_shape)
            return conv2 + cropped_inputs
