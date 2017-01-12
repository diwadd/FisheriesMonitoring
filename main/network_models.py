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

    def network_model_reg_small(self, rfx, rfy, channels, dropout_list):
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

        self.epsilon = gv.EPSILON_BN

        iw = int(rfx * gv.DEFAULT_IMAGE_WIDTH)
        ih = int(rfy * gv.DEFAULT_IMAGE_HEIGHT)

        self.rfx = rfx
        self.rfy = rfy
        self.width = iw
        self.height = ih
        self.channels = channels
        self.dropout_list = dropout_list
        self.dropout_one = tf.placeholder(tf.float32)
        self.dropout_two = tf.placeholder(tf.float32)
        self.dropout_three = tf.placeholder(tf.float32)
        self.dropout_four = tf.placeholder(tf.float32)
        self.dropout_five = tf.placeholder(tf.float32)

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


    def network_model_reg_small_xavier(self, rfx, rfy, channels, dropout_list):
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

        self.epsilon = gv.EPSILON_BN

        iw = int(rfx * gv.DEFAULT_IMAGE_WIDTH)
        ih = int(rfy * gv.DEFAULT_IMAGE_HEIGHT)

        self.rfx = rfx
        self.rfy = rfy
        self.width = iw
        self.height = ih
        self.channels = channels
        self.dropout_list = dropout_list
        self.dropout_one = tf.placeholder(tf.float32)
        self.dropout_two = tf.placeholder(tf.float32)
        self.dropout_three = tf.placeholder(tf.float32)
        self.dropout_four = tf.placeholder(tf.float32)
        self.dropout_five = tf.placeholder(tf.float32)

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

        #self.network_output = tf.nn.sigmoid(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)
        #self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected

        #self.temp_const = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="temp_const")
        #temp = tf.nn.relu(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)

        #self.network_output = tf.select(tf.greater_equal(temp, self.temp_const), self.temp_const, temp)


    def network_model_cla_small_xavier(self, rfx, rfy, channels, dropout_list):
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

        self.epsilon = gv.EPSILON_BN

        iw = int(rfx * gv.DEFAULT_IMAGE_WIDTH)
        ih = int(rfy * gv.DEFAULT_IMAGE_HEIGHT)

        self.rfx = rfx
        self.rfy = rfy
        self.width = iw
        self.height = ih
        self.channels = channels
        self.dropout_list = dropout_list
        self.dropout_one = tf.placeholder(tf.float32)
        self.dropout_two = tf.placeholder(tf.float32)
        self.dropout_three = tf.placeholder(tf.float32)
        self.dropout_four = tf.placeholder(tf.float32)
        self.dropout_five = tf.placeholder(tf.float32)

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
        #self.network_output = tf.nn.sigmoid(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)
        self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected

        self.temp_const = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="temp_const")
        temp = tf.nn.relu(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)

        self.network_output = tf.select(tf.greater_equal(temp, self.temp_const), self.temp_const, temp)





    def __init__(self, network_type, rfx, rfy, channels, dropout_list):

        self.sess = None
        self.saver_loader = None

        self.increment_constant = tf.constant(1, dtype=tf.float32, name="increment_constant")
        self.epoch_step_number = tf.Variable(0, dtype=tf.float32, name="epoch_step_number")

        self.epoch_step_number = tf.assign(self.epoch_step_number,
                                           tf.add(self.epoch_step_number, self.increment_constant))

        if network_type == "network_model_reg_small":
            self.network_type = network_type
            self.network_model_reg_small(rfx, rfy, channels, dropout_list)
        elif network_type == "network_model_reg_small_xavier":
            self.network_type = network_type
            self.network_model_reg_small_xavier(rfx, rfy, channels, dropout_list)
        else:
            pass


    def setup_loss(self, mini_batch_size):

        if (self.network_type == "network_model_reg_small" or
            self.network_type == "network_model_reg_small_xavier"):
            self.network_expected_output = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="network_expected_output")

            s = tf.subtract(self.network_output, self.network_expected_output)
            self.C = tf.reduce_sum(tf.multiply(s, s))
            m = tf.constant(2.0 * mini_batch_size, dtype=tf.float32)
            self.C = tf.divide(self.C, m)
        else:
            pass


    def setup_minimize(self, learning_rate):

        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.C)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.C)



    def setup_session(self, mode, network_model_file_name):

        if mode == "training":
            self.saver_loader = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
        elif mode == "training_continuation":
            self.saver_loader = tf.train.Saver()
            self.sess = tf.Session()

            self.saver_loader.restore(self.sess, network_model_file_name)
        else:
            sys.exit("ERROR: Choose a correct mode.")

    def train(self,
              x_train,
              x_valid,
              n_epochs,
              mini_batch_size,
              learning_rate,
              mode,
              network_model_file_name):

        self.setup_loss(mini_batch_size)
        self.setup_minimize(learning_rate)

        self.setup_session(mode, network_model_file_name)


        n_batches_per_epoch = int(len(x_train) / mini_batch_size)

        print("Training...")
        print("epoch_step_number: " + str(self.sess.run(self.epoch_step_number)))
        print("Number of mini batches: " + str(n_batches_per_epoch))
        for epoch in range(n_epochs):
            ptr = 0
            for batch in range(n_batches_per_epoch):
                start = time.time()

                mini_batch = x_train[ptr:ptr + mini_batch_size]
                images, labels = gv.read_image_chunk_real_labels(mini_batch, self.rfx, self.rfy)

                #print("image shape: " + str(images[0].shape))

                ptr = ptr + mini_batch_size

                parameter_dict = {self.network_input: images,
                                  self.network_expected_output: labels,
                                  self.dropout_one: self.dropout_list[0],
                                  self.dropout_two: self.dropout_list[1],
                                  self.dropout_three: self.dropout_list[2],
                                  self.dropout_four: self.dropout_list[3],
                                  self.dropout_five: self.dropout_list[4]}
                (self.train_step).run(session=self.sess, feed_dict=parameter_dict)

                c_val_train = (self.C).eval(session=self.sess, feed_dict=parameter_dict)
                stop = time.time()

                print("c_val_train value: %10s time: %10s" % (str(c_val_train), str(stop - start)))
                #print("time: %10s" % (str(stop - start)))

                #self.sess.run(self.epoch_step_number)
                self.saver_loader.save(self.sess, network_model_file_name)

                print("epoch_step_number: " + str(self.epoch_step_number.eval(session=self.sess)))

            images, labels = gv.read_image_chunk_real_labels(x_valid, self.rfx, self.rfy)
            parameter_dict = {self.network_input: images,
                              self.network_expected_output: labels,
                              self.dropout_one: self.dropout_list[0],
                              self.dropout_two: self.dropout_list[1],
                              self.dropout_three: self.dropout_list[2],
                              self.dropout_four: self.dropout_list[3],
                              self.dropout_five: self.dropout_list[4]}
            c_val_valid = (self.C).eval(session=self.sess, feed_dict=parameter_dict)
            print("c_val_valid value: " + str(c_val_valid))






# Some spare code

#self.network_output = tf.nn.sigmoid(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)
#self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected

#self.temp_const = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="temp_const")
#temp = tf.nn.relu(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)

#self.network_output = tf.select(tf.greater_equal(temp, self.temp_const), self.temp_const, temp)

