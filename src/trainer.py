"""
    Author: Mohamed K. Eid (mohamedkeid@gmail.com)
    Description: trainer class for training a new generative model
"""

import custom_vgg16 as vgg16
import helpers
import numpy as np
import os
import tensorflow as tf
import time
import urllib
import zipfile


class Trainer:
    def __init__(self, session, net, train_path, train_dims, print_training_status=True, print_every_n=100):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.net = net
        self.paths = {
            'out_dir': self.current_path + '/../output/',
            'style_file': train_path,
            'trained_generators_dir': self.current_path + '/../lib/generators/',
            'training_dir': self.current_path + '/../lib/train2014/',
            'training_url': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
        }
        self.session = session
        self.train_height = train_dims['height']
        self.train_width = train_dims['width']
        self.print_training_status = print_training_status
        self.train_n = print_every_n

    def train(self, epochs, learning_rate, content_layer, content_weight, style_layers, style_weight, tv_weight):
        # Check if there is training data available and initialize generator network
        self.__check_for_examples()

        # Initialize and process images and placeholders to be used for our descriptors
        art, art_shape = helpers.load_img_to(self.paths['style_file'], height=self.train_height, width=self.train_width)
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
                content_loss = helpers.get_content_loss(variable_model, content_model, content_layer) * content_weight

            if style_weight is 0:
                style_loss = tf.constant(0.)
            else:
                style_loss = helpers.get_style_loss(variable_model, style_model, style_layers) * style_weight

            if tv_weight is 0:
                tv_loss = tf.constant(0.)
            else:
                tv_loss = helpers.get_total_variation(variable_img, art_shape) * tv_weight

            total_loss = content_loss + style_loss + tv_loss

        # Optimization ops
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            grads = optimizer.compute_gradients(total_loss, trainable_vars)
            update_weights = optimizer.apply_gradients(grads)

        # Populate the training data
        print("Initializing session and loading training images..")
        example = self.__next_example(height=art_shape[1], width=art_shape[2])
        self.session.run(tf.global_variables_initializer())

        # Initialize threads and begin training
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

        # Alert that training has been completed and print the run time
        elapsed = time.time() - start_time
        print("Training complete. The session took %.2f seconds to complete." % elapsed)
        coord.request_stop()
        coord.join(threads)

        self.__save_model(trainable_vars)

    # Checks for training data to see if it's missing or not. Asks to download if missing.
    def __check_for_examples(self):
        # Asks on stdout to download MSCOCO data. Downloads if response is 'y'
        def ask_to_download():
            print("You've requested to train a new model. However, you've yet to download the training data.")

            answer = 0
            while answer is not 'y' and answer is not 'N':
                answer = input("Would you like to download the 13 GB file? [y/N] ").replace(" ", "")

            # Download weights if yes, else exit the program
            if answer == 'y':
                print("Downloading from %s. Please be patient..." % self.paths['training_url'])

                zip_save_path = self.current_path + '/../lib/images/train2014.zip'
                urllib.request.urlretrieve(self.paths['training_url'], zip_save_path)
                ask_to_unzip(zip_save_path)
            elif answer == 'N':
                self.__exit()

        # Asks on stdout to unzip a given zip file path. Unizips if response is 'y'
        def ask_to_unzip(path):
            answer = 0
            while answer is not 'y' and answer is not 'N':
                answer = input("The application requires the file to be unzipped. Unzip? [y/N] ").replace(" ", "")

            if answer == 'y':
                if not os.path.isdir(self.paths['training_dir']):
                    os.makedirs(self.paths['training_dir'])

                print("Unzipping file..")
                zip_ref = zipfile.ZipFile(path, 'r')
                zip_ref.extractall(self.current_path + '/../lib/')
                zip_ref.close()
                os.remove(path)
            else:
                self.__exit(0, message="Please unzip the program manually to run the program. Exiting..")

        # Ask to unzip training data if a previous attempt was made
        zip_path = os.path.abspath(self.current_path + '/../lib/images/train2014.zip')
        if os.path.isfile(zip_path):
            ask_to_unzip(zip_path)

        # Ask to download training data if the training dir does not exist or does not contain the needed files
        if not os.path.isdir(self.paths['training_dir']):
            ask_to_download()
        else:
            training_files = os.listdir(self.paths['training_dir'])
            num_training_files = len(training_files)
            if num_training_files <= 1:
                ask_to_download()

                # Retrieves next example image from queue

    # Returns a new training example
    def __next_example(self, height, width):
        filenames = tf.train.match_filenames_once(self.paths['training_dir'] + '*.jpg')
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, files = reader.read(filename_queue)
        training_img = tf.image.decode_jpeg(files, channels=3)
        training_img = tf.image.resize_images(training_img, [height, width])
        return training_img

    # Saves the weights with the name of the references style so that the net may stylize future images
    def __save_model(self, variables):
        print("Proceeding to save weights..")
        name = os.path.basename(self.paths['style_file']).replace('.jpg', '')
        gen_dir = self.paths['trained_generators_dir'] + name + '/'
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)
        saver = tf.train.Saver(variables)
        saver.save(self.session, gen_dir + name)

    def __exit(self, rc=0, message="Exiting the program.."):
        print(message)
        self.session.close()
        exit(rc)
