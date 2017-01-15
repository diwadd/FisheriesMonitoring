import time
import sys

import numpy as np
import tensorflow as tf

import global_variable as gv

# Set Tensorflow seed so we have reproducible results.
tf.set_random_seed(gv.TENSORFLOW_RANDOM_STATE)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.005, dtype=tf.float32)
  return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
  initial = tf.constant(0.005, shape=shape, dtype=tf.float32)
  return tf.Variable(initial, dtype=tf.float32)


class NeuralNetworkFishDetection:

    def network_model_reg_small(self):
        """
        This is a regression type network with the following structure:

        conv_layer_one_dropped:    (?, 180, 320,  8)
        conv_layer_two_dropped:    (?,  90, 160, 16)
        conv_layer_three_dropped:  (?,  90, 160, 16)
        conv_layer_four_dropped:   (?,  45,  80, 16)
        conv_layer_five_dropped:   (?,  23,  40, 16)


        :param rfx: scaling factor in the x direction
        :param rfy: scaling factor in the x direction
        :param channels: number of channels in the input image
        :param dropout_list: array of dropout probabilities for each layer
        :return: none
        """

        self.network_input = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels], name="network_input")
        self.network_output = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="network_output")

        kernel_one = weight_variable([5, 5, 3, 8])
        bias_one = bias_variable([8])

        conv_one = tf.nn.conv2d(self.network_input, kernel_one, strides=[1, 1, 1, 1], padding='SAME') + bias_one
        mean_one, vari_one = tf.nn.moments(conv_one, [0])
        scale_one = tf.Variable(tf.ones([8], dtype=tf.float32), dtype=tf.float32)
        beta_one = tf.Variable(tf.zeros([8], dtype=tf.float32), dtype=tf.float32)
        bn_one = tf.nn.batch_normalization(conv_one, mean_one, vari_one, beta_one, scale_one, self.epsilon)

        conv_layer_one = tf.nn.relu(bn_one)
        conv_layer_one_dropped = tf.nn.dropout(conv_layer_one, self.dropout_one)

        print("conv_layer_one_dropped: " + str((conv_layer_one_dropped).get_shape()))

        kernel_two = weight_variable([5, 5, 8, 16])
        bias_two = bias_variable([16])

        conv_two = tf.nn.conv2d(conv_layer_one_dropped, kernel_two, strides=[1, 2, 2, 1], padding='SAME') + bias_two
        mean_two, vari_two = tf.nn.moments(conv_two, [0])
        scale_two = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_two = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_two = tf.nn.batch_normalization(conv_two, mean_two, vari_two, beta_two, scale_two, self.epsilon)

        conv_layer_two = tf.nn.relu(bn_two)
        conv_layer_two_dropped = tf.nn.dropout(conv_layer_two, self.dropout_two)

        print("conv_layer_two_dropped: " + str((conv_layer_two_dropped).get_shape()))

        kernel_three = weight_variable([5, 5, 16, 16])
        bias_three = bias_variable([16])

        conv_three = tf.nn.conv2d(conv_layer_two_dropped, kernel_three, strides=[1, 1, 1, 1], padding='SAME') + bias_three
        mean_three, vari_three = tf.nn.moments(conv_three, [0])
        scale_three = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_three = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_three = tf.nn.batch_normalization(conv_three, mean_three, vari_three, beta_three, scale_three, self.epsilon)

        conv_layer_three = tf.nn.relu(bn_three)
        conv_layer_three_dropped = tf.nn.dropout(conv_layer_three, self.dropout_three)

        print("conv_layer_three_dropped: " + str((conv_layer_three_dropped).get_shape()))

        kernel_four = weight_variable([5, 5, 16, 16])
        bias_four = bias_variable([16])

        conv_four = tf.nn.conv2d(conv_layer_three_dropped, kernel_four, strides=[1, 2, 2, 1], padding='SAME') + bias_four
        mean_four, vari_four = tf.nn.moments(conv_four, [0])
        scale_four = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_four = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_four = tf.nn.batch_normalization(conv_four, mean_four, vari_four, beta_four, scale_four, self.epsilon)

        conv_layer_four = tf.nn.relu(bn_four)
        conv_layer_four_dropped = tf.nn.dropout(conv_layer_four, self.dropout_four)

        print("conv_layer_four_dropped: " + str((conv_layer_four_dropped).get_shape()))

        kernel_five = weight_variable([5, 5, 16, 16])
        bias_five = bias_variable([16])

        conv_five = tf.nn.conv2d(conv_layer_four_dropped, kernel_five, strides=[1, 2, 2, 1], padding='SAME') + bias_five
        mean_five, vari_five = tf.nn.moments(conv_five, [0])
        scale_five = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_five = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_five = tf.nn.batch_normalization(conv_five, mean_five, vari_five, beta_five, scale_five, self.epsilon)

        conv_layer_five = tf.nn.relu(bn_five)
        conv_layer_five_dropped = tf.nn.dropout(conv_layer_five, self.dropout_five)

        print("conv_layer_five_dropped: " + str((conv_layer_five_dropped).get_shape()))

        # Fully connected layer
        conv_layer_five_dropped_array = tf.reshape(conv_layer_five_dropped, [-1, 23 * 40 * 16])
        w_first_connected = weight_variable([23 * 40 * 16, 9 * 4])
        bias_first_connected = bias_variable([9 * 4])

        # Network output
        self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected
        self.network_expected_output = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="network_expected_output")


    def network_model_reg_small_xavier(self):
        """
        This is a regression type network with the following structure:

        conv_layer_one_dropped:    (?, 180, 320,  8)
        conv_layer_two_dropped:    (?,  90, 160, 16)
        conv_layer_three_dropped:  (?,  90, 160, 16)
        conv_layer_four_dropped:   (?,  45,  80, 16)
        conv_layer_five_dropped:   (?,  23,  40, 16)

        This function is the same as:
        - network_model_reg_small(self, rfx, rfy, channels, dropout_list)

        but also performs a Xavier initialization of weights.

        :param rfx: scaling factor in the x direction
        :param rfy: scaling factor in the x direction
        :param channels: number of channels in the input image
        :param dropout_list: array of dropout probabilities for each layer
        :return: none
        """

        self.network_input = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels], name="network_input")
        self.network_output = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="network_output")

        kernel_one = tf.get_variable("kernel_one",
                                     shape=[5, 5, 3, 8],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_one = bias_variable([8])

        conv_one = tf.nn.conv2d(self.network_input, kernel_one, strides=[1, 1, 1, 1], padding='SAME') + bias_one
        mean_one, vari_one = tf.nn.moments(conv_one, [0])
        scale_one = tf.Variable(tf.ones([8], dtype=tf.float32), dtype=tf.float32)
        beta_one = tf.Variable(tf.zeros([8], dtype=tf.float32), dtype=tf.float32)
        bn_one = tf.nn.batch_normalization(conv_one, mean_one, vari_one, beta_one, scale_one, self.epsilon)

        conv_layer_one = tf.nn.relu(bn_one)
        conv_layer_one_dropped = tf.nn.dropout(conv_layer_one, self.dropout_one)

        print("conv_layer_one_dropped: " + str((conv_layer_one_dropped).get_shape()))


        kernel_two = tf.get_variable("kernel_two",
                                     shape=[5, 5, 8, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_two = bias_variable([16])

        conv_two = tf.nn.conv2d(conv_layer_one_dropped, kernel_two, strides=[1, 2, 2, 1], padding='SAME') + bias_two
        mean_two, vari_two = tf.nn.moments(conv_two, [0])
        scale_two = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_two = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_two = tf.nn.batch_normalization(conv_two, mean_two, vari_two, beta_two, scale_two, self.epsilon)

        conv_layer_two = tf.nn.relu(bn_two)
        conv_layer_two_dropped = tf.nn.dropout(conv_layer_two, self.dropout_two)

        print("conv_layer_two_dropped: " + str((conv_layer_two_dropped).get_shape()))


        kernel_three = tf.get_variable("kernel_three",
                                     shape=[5, 5, 16, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_three = bias_variable([16])

        conv_three = tf.nn.conv2d(conv_layer_two_dropped, kernel_three, strides=[1, 1, 1, 1], padding='SAME') + bias_three
        mean_three, vari_three = tf.nn.moments(conv_three, [0])
        scale_three = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_three = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_three = tf.nn.batch_normalization(conv_three, mean_three, vari_three, beta_three, scale_three, self.epsilon)

        conv_layer_three = tf.nn.relu(bn_three)
        conv_layer_three_dropped = tf.nn.dropout(conv_layer_three, self.dropout_three)

        print("conv_layer_three_dropped: " + str((conv_layer_three_dropped).get_shape()))


        kernel_four = tf.get_variable("kernel_four",
                                     shape=[5, 5, 16, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_four = bias_variable([16])

        conv_four = tf.nn.conv2d(conv_layer_three_dropped, kernel_four, strides=[1, 2, 2, 1], padding='SAME') + bias_four
        mean_four, vari_four = tf.nn.moments(conv_four, [0])
        scale_four = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_four = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_four = tf.nn.batch_normalization(conv_four, mean_four, vari_four, beta_four, scale_four, self.epsilon)

        conv_layer_four = tf.nn.relu(bn_four)
        conv_layer_four_dropped = tf.nn.dropout(conv_layer_four, self.dropout_four)

        print("conv_layer_four_dropped: " + str((conv_layer_four_dropped).get_shape()))

        kernel_five = tf.get_variable("kernel_five",
                                     shape=[5, 5, 16, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_five = bias_variable([16])

        conv_five = tf.nn.conv2d(conv_layer_four_dropped, kernel_five, strides=[1, 2, 2, 1], padding='SAME') + bias_five
        mean_five, vari_five = tf.nn.moments(conv_five, [0])
        scale_five = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_five = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_five = tf.nn.batch_normalization(conv_five, mean_five, vari_five, beta_five, scale_five, self.epsilon)

        conv_layer_five = tf.nn.relu(bn_five)
        conv_layer_five_dropped = tf.nn.dropout(conv_layer_five, self.dropout_five)

        print("conv_layer_five_dropped: " + str((conv_layer_five_dropped).get_shape()))

        # Fully connected layer
        conv_layer_five_dropped_array = tf.reshape(conv_layer_five_dropped, [-1, 23 * 40 * 16])

        w_first_connected = tf.get_variable("w_first_connected",
                                     shape=[23 * 40 * 16, 9 * 4],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_first_connected = bias_variable([9 * 4])

        # Network output
        self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected
        self.network_expected_output = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="network_expected_output")


    def network_model_cla_small_xavier(self):
        """
        This is a classification type network with the following structure:

        conv_layer_one_dropped:    (?, 180, 320,  8)
        conv_layer_two_dropped:    (?,  90, 160, 16)
        conv_layer_three_dropped:  (?,  90, 160, 16)
        conv_layer_four_dropped:   (?,  45,  80, 16)
        conv_layer_five_dropped:   (?,  23,  40, 16)

        This function constructs a neural network that:
        - performs a Xavier initialization of weights.
        - binarizes the result and uses a classification scheme instead of
          regression which is more appropriate.

        :param rfx: scaling factor in the x direction
        :param rfy: scaling factor in the x direction
        :param channels: number of channels in the input image
        :param dropout_list: array of dropout probabilities for each layer
        :return: none
        """

        self.network_input = tf.placeholder(tf.float32, shape=[None, self.height, self.width, self.channels], name="network_input")

        # We have self.n_bins + 1 to mark slots that don't have rectangles.
        self.network_output = tf.placeholder(tf.float32, shape=[None, 9 * 4 * (self.n_bins + 1)], name="network_output")

        kernel_one = tf.get_variable("kernel_one",
                                     shape=[5, 5, 3, 8],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_one = bias_variable([8])

        conv_one = tf.nn.conv2d(self.network_input, kernel_one, strides=[1, 1, 1, 1], padding='SAME') + bias_one
        mean_one, vari_one = tf.nn.moments(conv_one, [0])
        scale_one = tf.Variable(tf.ones([8], dtype=tf.float32), dtype=tf.float32)
        beta_one = tf.Variable(tf.zeros([8], dtype=tf.float32), dtype=tf.float32)
        bn_one = tf.nn.batch_normalization(conv_one, mean_one, vari_one, beta_one, scale_one, self.epsilon)

        conv_layer_one = tf.nn.relu(bn_one)
        conv_layer_one_dropped = tf.nn.dropout(conv_layer_one, self.dropout_one)

        print("conv_layer_one_dropped: " + str((conv_layer_one_dropped).get_shape()))


        kernel_two = tf.get_variable("kernel_two",
                                     shape=[5, 5, 8, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_two = bias_variable([16])

        conv_two = tf.nn.conv2d(conv_layer_one_dropped, kernel_two, strides=[1, 2, 2, 1], padding='SAME') + bias_two
        mean_two, vari_two = tf.nn.moments(conv_two, [0])
        scale_two = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_two = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_two = tf.nn.batch_normalization(conv_two, mean_two, vari_two, beta_two, scale_two, self.epsilon)

        conv_layer_two = tf.nn.relu(bn_two)
        conv_layer_two_dropped = tf.nn.dropout(conv_layer_two, self.dropout_two)

        print("conv_layer_two_dropped: " + str((conv_layer_two_dropped).get_shape()))


        kernel_three = tf.get_variable("kernel_three",
                                     shape=[5, 5, 16, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_three = bias_variable([16])

        conv_three = tf.nn.conv2d(conv_layer_two_dropped, kernel_three, strides=[1, 1, 1, 1], padding='SAME') + bias_three
        mean_three, vari_three = tf.nn.moments(conv_three, [0])
        scale_three = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_three = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_three = tf.nn.batch_normalization(conv_three, mean_three, vari_three, beta_three, scale_three, self.epsilon)

        conv_layer_three = tf.nn.relu(bn_three)
        conv_layer_three_dropped = tf.nn.dropout(conv_layer_three, self.dropout_three)

        print("conv_layer_three_dropped: " + str((conv_layer_three_dropped).get_shape()))


        kernel_four = tf.get_variable("kernel_four",
                                     shape=[5, 5, 16, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_four = bias_variable([16])

        conv_four = tf.nn.conv2d(conv_layer_three_dropped, kernel_four, strides=[1, 2, 2, 1], padding='SAME') + bias_four
        mean_four, vari_four = tf.nn.moments(conv_four, [0])
        scale_four = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_four = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_four = tf.nn.batch_normalization(conv_four, mean_four, vari_four, beta_four, scale_four, self.epsilon)

        conv_layer_four = tf.nn.relu(bn_four)
        conv_layer_four_dropped = tf.nn.dropout(conv_layer_four, self.dropout_four)

        print("conv_layer_four_dropped: " + str((conv_layer_four_dropped).get_shape()))

        kernel_five = tf.get_variable("kernel_five",
                                     shape=[5, 5, 16, 16],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_five = bias_variable([16])

        conv_five = tf.nn.conv2d(conv_layer_four_dropped, kernel_five, strides=[1, 2, 2, 1], padding='SAME') + bias_five
        mean_five, vari_five = tf.nn.moments(conv_five, [0])
        scale_five = tf.Variable(tf.ones([16], dtype=tf.float32), dtype=tf.float32)
        beta_five = tf.Variable(tf.zeros([16], dtype=tf.float32), dtype=tf.float32)
        bn_five = tf.nn.batch_normalization(conv_five, mean_five, vari_five, beta_five, scale_five, self.epsilon)

        conv_layer_five = tf.nn.relu(bn_five)
        conv_layer_five_dropped = tf.nn.dropout(conv_layer_five, self.dropout_five)

        print("conv_layer_five_dropped: " + str((conv_layer_five_dropped).get_shape()))

        # Fully connected layer
        conv_layer_five_dropped_array = tf.reshape(conv_layer_five_dropped, [-1, 23 * 40 * 16])

        # We have self.n_bins + 1 to mark slots that don't have rectangles.
        w_first_connected = tf.get_variable("w_first_connected",
                                     shape=[23 * 40 * 16, 9 * 4 * (self.n_bins + 1)],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_first_connected = bias_variable([9 * 4 * (self.n_bins + 1)])

        # Network output
        self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected
        self.network_expected_output = tf.placeholder(tf.float32, shape=[None, 9 * 4 * (self.n_bins + 1)], name="network_expected_output")

        self.network_output_for_prediction = tf.nn.softmax(self.network_output)




    def __init__(self, network_type, rfx, rfy, channels, dropout_list, n_bins=256):

        iw = int(rfx * gv.DEFAULT_IMAGE_WIDTH)
        ih = int(rfy * gv.DEFAULT_IMAGE_HEIGHT)

        self.width = iw
        self.height = ih
        self.rfx = rfx
        self.rfy = rfy

        self.epsilon = gv.EPSILON_BN
        self.channels = channels

        self.n_bins = n_bins

        self.binning_array_width = None
        self.binning_array_height = None

        self.sess = None
        self.saver_loader = None

        self.increment_constant = tf.constant(1, dtype=tf.float32, name="increment_constant")
        self.control_step_number = tf.Variable(0, dtype=tf.float32, name="epoch_step_number")

        self.control_step_number = tf.assign(self.control_step_number,
                                           tf.add(self.control_step_number, self.increment_constant))

        self.dropout_list = dropout_list
        self.dropout_one = tf.placeholder(tf.float32)
        self.dropout_two = tf.placeholder(tf.float32)
        self.dropout_three = tf.placeholder(tf.float32)
        self.dropout_four = tf.placeholder(tf.float32)
        self.dropout_five = tf.placeholder(tf.float32)
        self.dropout_six = tf.placeholder(tf.float32)
        self.dropout_seven = tf.placeholder(tf.float32)
        self.dropout_eight = tf.placeholder(tf.float32)

        NUMBER_OF_DROPOUT_PLACEHOLDERS = 8
        if len(self.dropout_list) < NUMBER_OF_DROPOUT_PLACEHOLDERS:
            n_missing_dropout_list_elements = NUMBER_OF_DROPOUT_PLACEHOLDERS - len(self.dropout_list)
            dropout_list_addition = [1.0 for i in range(n_missing_dropout_list_elements)]
            self.dropout_list = self.dropout_list + dropout_list_addition

        self.regression_network = False
        self.classification_network = False

        if network_type == "network_model_reg_small":
            self.network_type = network_type
            self.network_model_reg_small()
            self.regression_network = True
        elif network_type == "network_model_reg_small_xavier":
            self.network_type = network_type
            self.network_model_reg_small_xavier()
            self.regression_network = True
        elif network_type == "network_model_cla_small_xavier":
            self.network_type = network_type
            self.network_model_cla_small_xavier()

            self.binning_array_width = [i * self.width / self.n_bins for i in range(self.n_bins)]
            self.binning_array_height = [i * self.height / self.n_bins for i in range(self.n_bins)]

            self.classification_network = True
        else:
            pass


        self.parameter_dict = {self.network_input: None,
                               self.network_expected_output: None,
                               self.dropout_one: self.dropout_list[0],
                               self.dropout_two: self.dropout_list[1],
                               self.dropout_three: self.dropout_list[2],
                               self.dropout_four: self.dropout_list[3],
                               self.dropout_five: self.dropout_list[4],
                               self.dropout_six: self.dropout_list[5],
                               self.dropout_seven: self.dropout_list[6],
                               self.dropout_eight: self.dropout_list[7]}


    def setup_loss(self, mini_batch_size):

        if self.regression_network == True:
            # Set you the chi2 like loss

            s = tf.subtract(self.network_output, self.network_expected_output)
            self.C = tf.reduce_sum(tf.multiply(s, s))
            m = tf.constant(2.0 * mini_batch_size, dtype=tf.float32)
            self.C = tf.divide(self.C, m)
        elif self.classification_network == True:
            self.C = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network_output, self.network_expected_output))
        else:
            pass


    def setup_minimize(self, initial_learning_rate, decay_steps, decay_rate):

        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                            self.global_step,
                                                            decay_steps,
                                                            decay_rate,
                                                            staircase=True)

        self.train_step = tf.train.AdamOptimizer(self.decaying_learning_rate).minimize(self.C, global_step=self.global_step)


    def setup_session(self, mode, network_model_file_name):
        """
        Chose whether you want to train from the begining or whether you what to
        continue with an already pre trained network.

        :param mode:
        :param network_model_file_name:
        :return:
        """

        if mode == "training":
            self.saver_loader = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
        elif mode == "training_continuation_or_prediction":
            self.saver_loader = tf.train.Saver()
            self.sess = tf.Session()

            self.saver_loader.restore(self.sess, network_model_file_name)
        else:
            sys.exit("ERROR: Choose a correct mode.")



    def uoi_and_loss_for_image_set(self, x_valid):
        """
        Calculate the chi2 loss and the uoi for and image set. Normally,
        the image set is large and will not fit into memory. This function splits
        the initial set of images into smaller sets and computes the chi2 loss and
        the uoi for these smaller sets. It then combines the results.
        For uoi it actually calculates the average.

        :param x_valid:
        :return:
        """
        N = len(x_valid)

        mini_batch_size = gv.MINI_BATCHES_FOR_LARGE_SET_PROCESSING
        n_batches = round(N / mini_batch_size)

        ptr = 0

        uoi_sum = 0.0
        uoi_images = 0.0
        total_loss = 0.0
        for n in range(n_batches):
            mini_batch = x_valid[ptr:ptr + mini_batch_size]
            ptr = ptr + mini_batch_size

            images = None
            labels = None
            if self.regression_network == True:
                images, labels = gv.read_image_chunk_real_labels(mini_batch, self.rfx, self.rfy)
            elif self.classification_network == True:
                images, labels, n_rects_per_img = gv.read_image_chunk_hist_labels(mini_batch, self.rfx, self.rfy, self.n_bins)
            else:
                pass

            self.parameter_dict[self.network_input] = images
            self.parameter_dict[self.network_expected_output] = labels
            predicted_rects = (self.network_output).eval(session=self.sess, feed_dict=self.parameter_dict)


            uoi_sum_batch = None
            uoi_images_batch = None
            if self.regression_network == True:
                uoi_sum_batch, uoi_images_batch = gv.uoi_for_set_of_labels(labels, predicted_rects)

            elif self.classification_network == True:
                uoi_sum_batch, uoi_images_batch = gv.uoi_for_set_of_labels_cla_version(labels,
                                                                                       predicted_rects,
                                                                                       self.binning_array_width,
                                                                                       self.binning_array_height,
                                                                                       n_rects_per_img)
            else:
                pass

            uoi_sum = uoi_sum + uoi_sum_batch
            uoi_images = uoi_images + uoi_images_batch

            loss = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)
            total_loss = total_loss + mini_batch_size*loss

        average_uoi = uoi_sum/uoi_images
        return average_uoi, total_loss/len(x_valid)



    def train(self,
              x_train,
              x_valid,
              n_epochs,
              mini_batch_size,
              initial_learning_rate,
              mode,
              network_model_file_name,
              decay_steps,
              decay_rate):
        """
        Train the selected network model.

        :param x_train:
        :param x_valid:
        :param n_epochs:
        :param mini_batch_size:
        :param initial_learning_rate:
        :param mode:
        :param network_model_file_name:
        :return:
        """

        self.setup_loss(mini_batch_size)
        self.setup_minimize(initial_learning_rate, decay_steps, decay_rate)

        self.setup_session(mode, network_model_file_name)


        n_batches_per_epoch = round(len(x_train) / mini_batch_size)

        print("Training...")
        print("epoch_step_number: " + str(self.sess.run(self.control_step_number)))
        print("Number of mini batches: " + str(n_batches_per_epoch))
        print("decaying_learning_rate: " + str(self.decaying_learning_rate.eval(session=self.sess)))
        for epoch in range(n_epochs):
            ptr = 0
            start = time.time()
            for batch in range(n_batches_per_epoch):
                mini_batch = x_train[ptr:ptr + mini_batch_size]
                ptr = ptr + mini_batch_size

                images = None
                labels = None
                if self.regression_network == True:
                    images, labels = gv.read_image_chunk_real_labels(mini_batch, self.rfx, self.rfy)
                elif self.classification_network == True:
                    images, labels, n_rects_per_img = gv.read_image_chunk_hist_labels(mini_batch, self.rfx, self.rfy, self.n_bins)
                else:
                    pass

                self.parameter_dict[self.network_input] = images
                self.parameter_dict[self.network_expected_output] = labels
                #no = self.network_output.eval(session=self.sess, feed_dict=self.parameter_dict)

                #print(no)


                (self.train_step).run(session=self.sess, feed_dict=self.parameter_dict)

                if (batch % 5 == 0):
                    c_val_train = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)
                    print("(in batch loop) c_val_train value: %10s" % (str(c_val_train)))

            stop = time.time()
            print("Epoch time: " + str(stop - start))

            self.saver_loader.save(self.sess, network_model_file_name)

            print("control_step_number: " + str(self.control_step_number.eval(session=self.sess)))
            print("Evaluating the validation set.")

            print("decaying_learning_rate: " + str(self.decaying_learning_rate.eval(session=self.sess)))
            print("global_step: " + str(self.global_step.eval(session=self.sess)))

            average_uoi, total_loss = self.uoi_and_loss_for_image_set(x_valid)

            print("Average uoi: %15s" % (str(average_uoi)))
            print("Total loss: %15s" % (str(total_loss)))

            #print("c_val_valid value: " + str(c_val_valid))



    def predict(self, x_test, network_model_file_name):
        """
        Use the trained network to make predictinos.

        :param x_test:
        :param network_model_file_name:
        :return:
        """

        self.setup_session("training_continuation_or_prediction", network_model_file_name)

        N = len(x_test)

        predicted_labels = [np.zeros((9, 4)) for i in range(N)]
        #predicted_labels = np.zeros((N, 9, 4))
        pl_index = 0

        mini_batch_size = gv.MINI_BATCHES_FOR_LARGE_SET_PROCESSING
        n_batches = round(N / mini_batch_size)

        ptr = 0
        for n in range(n_batches):
            mini_batch = x_test[ptr:ptr + mini_batch_size]
            images, labels = gv.read_image_chunk_real_labels(mini_batch, self.rfx, self.rfy)

            ptr = ptr + mini_batch_size

            #print("Processed %15s mini_batches o validation set out of %15s" % (str(n), str(n_batches)), end="\r")

            self.parameter_dict[self.network_input] = images
            self.parameter_dict[self.network_expected_output] = labels

            predicted_labels_mini_batch = None
            if self.regression_network == True:
                predicted_labels_mini_batch = (self.network_output).eval(session=self.sess,
                                                                         feed_dict=self.parameter_dict)
            elif self.classification_network == True:
                predicted_labels_mini_batch = (self.network_output_for_prediction).eval(session=self.sess,
                                                                         feed_dict=self.parameter_dict)
            else:
                pass



            for j in range(len(predicted_labels_mini_batch)):
                predicted_labels[pl_index] = predicted_labels_mini_batch[j].reshape((9,4))
                pl_index = pl_index + 1


        return predicted_labels






# Some spare code

#self.network_output = tf.nn.sigmoid(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)
#self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected

#self.temp_const = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="temp_const")
#temp = tf.nn.relu(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)

#self.network_output = tf.select(tf.greater_equal(temp, self.temp_const), self.temp_const, temp)

