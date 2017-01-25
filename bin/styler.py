import custom_vgg16 as vgg16
import math_helpers
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import time
import urllib
import zipfile
from scipy.misc import toimage


class Styler:
    def __init__(self, session, net):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.examples = []
        self.net = net
        self.paths = {
            'out_dir': self.current_path + '/../output/',
            'style_file': self.current_path + '/../lib/images/style/great-wave-of-kanagawa.jpg',
            'trained_generators_dir': self.current_path + '/../lib/generators/',
            'training_dir': self.current_path + '/../lib/images/train2014/',
            'training_url': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
        }
        self.session = session
        self.train_height = 400
        self.train_width = 400
        self.print_training_status = True
        self.train_n = 100

    # Checks for training data to see if it's missing or not. Asks to download if missing.
    def check_for_examples(self):

        def ask_to_download():
            answer = 0
            while answer is not 'y' and answer is not 'N':
                answer = input("Would you like to download the 13 GB file? [y/N] ").replace(" ", "")

            # Download weights if yes, else exit the program
            if answer == 'y':
                print("Downloading from %s. Please be patient..." % self.paths['training_url'])

                urllib.request.urlretrieve(self.paths['training_dir'], 'train2014.zip')
                ask_to_unzip(self.paths['training'] + 'train2014.zip')
            elif answer == 'N':
                print("Exiting the program..")
                exit(0)

        # Asks on stdout to unzip a given zip file path. Unizips if response is 'y'
        def ask_to_unzip(path):
            answer = 0
            while answer is not 'y' and answer is not 'N':
                answer = input("The application requires the file to be unzipped. Unzip? [y/N] ").replace(" ", "")

            if answer == 'y':
                print("Unzipping file..")
                zip_ref = zipfile.ZipFile(path, 'r')
                zip_ref.extractall(self.paths['training_dir'])
                zip_ref.close()
            else:
                print("Please unzip the program manually to run the program. Exiting..")
                exit(0)

        training_files = os.listdir(self.paths['training_dir'])
        num_training_files = len(training_files)

        if num_training_files <= 1:
            print("Training data could not be found.")
            ask_to_download()

    # Returns a numpy array of an image specified by its path
    def load_img(self, path):
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
    def load_img_to(self, path, height=None, width=None):
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

    # Loads the generator net's weights for producing the style
    def load_style(self, style):
        saver = tf.train.import_meta_graph(self.paths['trained_generators_dir'] + style + '.meta')
        saver.restore(self.session, self.paths['trained_generators_dir'])

    # Retrieves next example image from queue
    def next_example(self, height, width):
        filenames = tf.train.match_filenames_once(self.paths['training_dir'] + '*.jpg')
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, files = reader.read(filename_queue)
        training_img = tf.image.decode_jpeg(files, channels=3)
        training_img = tf.image.resize_images(training_img, [height, width])
        return training_img

    # Renders the generated image given a tensorflow session and a variable image (x)
    def render(self, x, feed_dict, display=False, path_out=None):
        shape = x.get_shape().as_list()
        img = self.session.run(x, feed_dict=feed_dict)
        clipped_img = np.clip(img, 0., 1.)

        if display:
            toimage(np.reshape(clipped_img, shape[1:])).show()

        if path_out:
            toimage(np.reshape(clipped_img, shape[1:])).save(path_out)

    def train(self, epochs, learning_rate, content_layer, content_weight, style_layers, style_weight, tv_weight):
        # Check if there is training data available and initialize generator network
        self.check_for_examples()

        # Initialize and process images and placeholders to be used for our descriptors
        art, art_shape = self.load_img_to(self.paths['style_file'], height=self.train_height, width=self.train_width)
        art_shape = [1] + art_shape
        art = art.reshape(art_shape).astype(np.float32)

        # Generator Network ops
        variable_placeholder = tf.placeholder(dtype=tf.float32, shape=art_shape)
        self.net.build(variable_placeholder)
        variable_img = self.net.output

        # VGG Network ops
        with tf.name_scope('vgg_style'):
            style_model = vgg16.Vgg16()
            style_model.build(art, shape=art_shape[1:])

        with tf.name_scope('vgg_content'):
            content_placeholder = tf.placeholder(dtype=tf.float32, shape=art_shape)
            content_model = vgg16.Vgg16()
            content_model.build(content_placeholder, shape=art_shape[1:])

        with tf.name_scope('vgg_variable'):
            variable_model = vgg16.Vgg16()
            variable_model.build(variable_img, shape=art_shape[1:])

        # Loss ops
        with tf.name_scope('loss'):
            if content_weight is 0:
                content_loss = tf.constant(0.)
            else:
                content_loss = math_helpers.get_content_loss(variable_model, content_model, content_layer) * content_weight

            if style_weight is 0:
                style_loss = tf.constant(0.)
            else:
                style_loss = math_helpers.get_style_loss(variable_model, style_model, style_layers) * style_weight

            if tv_weight is 0:
                tv_loss = tf.constant(0.)
            else:
                tv_loss = math_helpers.get_total_variation(variable_img, art_shape) * tv_weight

            total_loss = content_loss + style_loss + tv_loss

        # Optimization ops
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            grads = optimizer.compute_gradients(total_loss, trainable_vars)
            update_weights = optimizer.apply_gradients(grads)

        # Populate the training data
        print("Initializing session and loading training images..")
        example = self.next_example(height=art_shape[1], width=art_shape[2])
        self.session.run(tf.initialize_all_variables())

        print("Begining training..")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()

        for i in range(epochs):
            self.net.is_training = True

            # Get next training image from batch and reshape it to include a batch size of 1
            training_img = self.session.run(example) / 255.
            training_img = training_img.reshape([1] + list(training_img.shape)).astype(np.float32)

            # Initialize new feed dict for the training iteration and invoke the update op
            feed_dict = {variable_placeholder: training_img, content_placeholder: training_img}
            _, loss = self.session.run([update_weights, total_loss], feed_dict=feed_dict)

            if self.print_training_status and i % self.train_n == 0:
                print("Epoch %06d | Loss %.06f" % (i, loss))
                in_path = self.current_path + '/../nyc.jpg'
                input_img, input_shape = self.load_img_to(in_path, height=self.train_height, width=self.train_width)
                input_img = input_img.reshape([1] + input_shape).astype(np.float32)
                path_out = self.current_path + '/../output/' + str(start_time) + '.jpg'
                self.render(variable_img, feed_dict={variable_placeholder: input_img}, path_out=path_out)

        # Alert that training has been completed and print the run time
        elapsed = time.time() - start_time
        print("Training complete. The session took %.2f seconds to complete." % elapsed)
        coord.request_stop()
        coord.join(threads)

        # Save the weights with the name of the references style so that the net may stylize future images
        print("Proceeding to save weights..")
        name = os.path.basename(self.paths['style_file']).replace('.jpg', '')
        os.makedirs(self.paths['trained_generators_dir'] + name)
        saver = tf.train.Saver(trainable_vars)
        saver.save(self.session, self.paths['trained_generators_dir'] + name)
