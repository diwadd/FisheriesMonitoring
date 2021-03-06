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

    def automatic_model_reg_xavier(self,
                                   shape_list,
                                   index_conv_layers,
                                   index_fully_conected_layers,
                                   network_type):


        print("Input image size: " + str(shape_list[0]))

        self.network_input = tf.placeholder(tf.float32,
                                            shape=[None,
                                                   shape_list[0][0],
                                                   shape_list[0][1],
                                                   shape_list[0][2]],
                                            name="network_input")

        # ----------------------------------------------------------
        # Example shape_list:
        # shape_list = [
        #               [120, 320, 3], # input image dimensions
        #               [[5, 5, 3, 8],[1, 1, 1, 1]], # conv layer
        #               [[5, 5, 3, 8],[1, 1, 1, 1]], # conv layer
        #               ...
        #               [23 * 40 * 8, 1024], # fully connected layer
        #               [1024, 9 * 4 * 128 + 1] # readout layer
        #               ]
        # ----------------------------------------------------------

        dropout_index = 0
        current_layer = self.network_input
        for i in range(index_conv_layers, index_fully_conected_layers):
            print("Creating conv layer: " + str(shape_list[i]))
            kernel = tf.get_variable("kernel_" + str(i),
                                      shape=shape_list[i][0],
                                      initializer=tf.contrib.layers.xavier_initializer())
            bias = bias_variable([shape_list[i][0][3]])
            conv = tf.nn.conv2d(current_layer,
                                    kernel,
                                    strides=shape_list[i][1],
                                    padding='SAME') + bias

            mean, vari = tf.nn.moments(conv, [0])
            scale = tf.Variable(tf.ones([shape_list[i][0][3]], dtype=tf.float32), dtype=tf.float32)
            beta = tf.Variable(tf.zeros([shape_list[i][0][3]], dtype=tf.float32), dtype=tf.float32)

            bn = tf.nn.batch_normalization(conv,
                                            mean,
                                            vari,
                                            beta,
                                            scale,
                                            self.epsilon)

            current_layer = tf.nn.relu(bn)
            print("Conv layer %6s, shape: %15s" % (str(i), str(current_layer.get_shape())))

        current_layer = tf.reshape(current_layer, [-1, shape_list[index_fully_conected_layers][0]])

        for j in range(index_fully_conected_layers, len(shape_list) - 1):
            print("Creating fully connected layer: " + str(shape_list[j]))
            w_connected = tf.get_variable("w_connected_" + str(j),
                                          shape=shape_list[j],
                                          initializer=tf.contrib.layers.xavier_initializer())
            bias_connected = bias_variable([shape_list[j][1]])
            fully = tf.matmul(current_layer, w_connected) + bias_connected

            mean_fully, vari_fully = tf.nn.moments(fully, [0])
            scale_fully = tf.Variable(tf.ones([shape_list[j][1]], dtype=tf.float32), dtype=tf.float32)
            beta_fully = tf.Variable(tf.zeros([shape_list[j][1]], dtype=tf.float32), dtype=tf.float32)

            bn_connected = tf.nn.batch_normalization(fully,
                                                 mean_fully,
                                                 vari_fully,
                                                 beta_fully,
                                                 scale_fully,
                                                 self.epsilon)

            fully_connected = tf.nn.relu(bn_connected)

            fully_connected_dropped = tf.nn.dropout(fully_connected,
                                                    self.dropout_variable_list[dropout_index],
                                                    name="fully_tf.nn.dropout_" + str(j))
            dropout_index = dropout_index + 1

            current_layer = fully_connected_dropped
            print("Fully connected layer %6s, shape: %15s" % (str(j), str(current_layer.get_shape())))

        # Network output
        print("Readout layer size: " + str(shape_list[-1]))
        w_readout = tf.get_variable("w_readout",
                                     shape=shape_list[-1],
                                     initializer=tf.contrib.layers.xavier_initializer())
        bias_readout = bias_variable([shape_list[-1][1]])

        if network_type == "classification":
            self.network_output = tf.matmul(current_layer, w_readout) + bias_readout
            self.network_expected_output = tf.placseholder(tf.float32,
                                                          shape=[None, shape_list[-1][1]],
                                                          name="network_expected_output")
            self.network_output_for_prediction = tf.nn.softmax(self.network_output)
        elif network_type == "regression":
            self.network_output = tf.matmul(current_layer, w_readout) + bias_readout
            self.network_expected_output = tf.placeholder(tf.float32,
                                                          shape=[None, shape_list[-1][1]],
                                                          name="network_expected_output")
            self.network_output_for_prediction = tf.nn.sigmoid(self.network_output)

        else:
            pass



    def __init__(self,
                 network_type,
                 dropout_list,
                 shape_list,
                 index_conv_layers,
                 index_fully_conected_layers,
                 nr_of_h_bins,
                 nr_of_w_bins):

        self.width = shape_list[0][1]
        self.height = shape_list[0][0]

        self.epsilon = gv.EPSILON_BN

        self.nr_of_h_bins = nr_of_h_bins
        self.nr_of_w_bins = nr_of_w_bins

        self.dropout_one = tf.placeholder(tf.float32)
        self.dropout_two = tf.placeholder(tf.float32)
        self.dropout_three = tf.placeholder(tf.float32)
        self.dropout_four = tf.placeholder(tf.float32)
        self.dropout_five = tf.placeholder(tf.float32)
        self.dropout_six = tf.placeholder(tf.float32)
        self.dropout_seven = tf.placeholder(tf.float32)
        self.dropout_eight = tf.placeholder(tf.float32)
        self.dropout_nine = tf.placeholder(tf.float32)

        self.dropout_list = dropout_list
        self.dropout_variable_list = [self.dropout_one,
                                      self.dropout_two,
                                      self.dropout_three,
                                      self.dropout_four,
                                      self.dropout_five,
                                      self.dropout_six,
                                      self.dropout_seven,
                                      self.dropout_eight,
                                      self.dropout_nine]

        NUMBER_OF_DROPOUT_PLACEHOLDERS = len(self.dropout_variable_list)
        if len(self.dropout_list) < NUMBER_OF_DROPOUT_PLACEHOLDERS:
            n_missing_dropout_list_elements = NUMBER_OF_DROPOUT_PLACEHOLDERS - len(self.dropout_list)
            dropout_list_addition = [1.0 for i in range(n_missing_dropout_list_elements)]
            self.dropout_list = self.dropout_list + dropout_list_addition

        print("Extended dropout list:")
        print(self.dropout_list)

        self.binning_array_width = None
        self.binning_array_height = None

        self.sess = None
        self.saver_loader = None


        self.increment_constant = tf.constant(1, dtype=tf.float32, name="increment_constant")
        self.control_step_number = tf.Variable(0, dtype=tf.float32, name="epoch_step_number")

        self.control_step_number = tf.assign(self.control_step_number,
                                           tf.add(self.control_step_number, self.increment_constant))

        self.regression_network = False
        self.classification_network = False

        if network_type == "regression":
            self.network_type = network_type
            self.automatic_model_reg_xavier(shape_list,
                                            index_conv_layers,
                                            index_fully_conected_layers,
                                            network_type)
            self.regression_network = True

        elif network_type == "classification":
            self.network_type = network_type
            self.automatic_model_reg_xavier(shape_list,
                                            index_conv_layers,
                                            index_fully_conected_layers,
                                            network_type)
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
                               self.dropout_eight: self.dropout_list[7],
                               self.dropout_nine: self.dropout_list[8]}

    def setup_loss(self, mini_batch_size):

        self.mini_batch_size = mini_batch_size
        if self.regression_network == True:
            # Set you the chi2 like loss

            #s = tf.subtract(self.network_output, self.network_expected_output)
            #self.C = tf.reduce_sum(tf.multiply(s, s))
            #m = tf.constant(2.0 * mini_batch_size, dtype=tf.float32)
            #self.C = tf.divide(self.C, m)

            self.C = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.network_expected_output,
                                                                            logits=self.network_output))

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

        self.train_step = tf.train.AdamOptimizer(self.decaying_learning_rate).minimize(self.C, 
                                                                                       global_step=self.global_step,
                                                                                       aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)


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



    def loss_for_image_set(self, x_valid):
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

        mini_batch_size = self.mini_batch_size
        n_batches = round(N / mini_batch_size)

        ptr = 0
        total_loss = 0.0
        for n in range(n_batches):
            mini_batch = x_valid[ptr:ptr + mini_batch_size]
            ptr = ptr + mini_batch_size

            images = None
            labels = None

            images, labels = gv.read_image_chunk_fish_mask(mini_batch,
                                                           self.height,
                                                           self.width,
                                                           self.nr_of_h_bins,
                                                           self.nr_of_w_bins,
                                                           self.network_type)

            images = gv.scale_image_by_255(images)

            self.set_parameter_dict_for_evaluation(images, labels)

            loss = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)
            #print("internal loss: " + str(loss))
            #print("mini_batch: " + str(mini_batch))
            total_loss = total_loss + (2.0*mini_batch_size)*loss

        return total_loss/(2.0*len(x_valid))


    def set_parameter_dict_for_train(self, images, labels):

        self.parameter_dict = {self.network_input: images,
                               self.network_expected_output: labels,
                               self.dropout_one: self.dropout_list[0],
                               self.dropout_two: self.dropout_list[1],
                               self.dropout_three: self.dropout_list[2],
                               self.dropout_four: self.dropout_list[3],
                               self.dropout_five: self.dropout_list[4],
                               self.dropout_six: self.dropout_list[5],
                               self.dropout_seven: self.dropout_list[6],
                               self.dropout_eight: self.dropout_list[7],
                               self.dropout_nine: self.dropout_list[8]}



    def set_parameter_dict_for_evaluation(self, images, labels):

        self.parameter_dict = {self.network_input: images,
                               self.network_expected_output: labels,
                               self.dropout_one: 1.0,
                               self.dropout_two: 1.0,
                               self.dropout_three: 1.0,
                               self.dropout_four: 1.0,
                               self.dropout_five: 1.0,
                               self.dropout_six: 1.0,
                               self.dropout_seven: 1.0,
                               self.dropout_eight: 1.0,
                               self.dropout_nine: 1.0}


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

        print("Untrained network:")
        total_loss = self.loss_for_image_set(x_valid)

        print("Total loss: %15s" % (str(total_loss)))

        print("\nTraining...")
        print("epoch_step_number: " + str(self.sess.run(self.control_step_number)))
        print("Number of mini batches: " + str(n_batches_per_epoch))
        print("decaying_learning_rate: " + str(self.decaying_learning_rate.eval(session=self.sess)))
        for epoch in range(n_epochs):
            ptr = 0

            train_loss = 0.0
            for batch in range(n_batches_per_epoch):
                start = time.time()
                mini_batch = x_train[ptr:ptr + mini_batch_size]
                ptr = ptr + mini_batch_size


                images, labels = gv.read_image_chunk_fish_mask(mini_batch,
                                                               self.height,
                                                               self.width,
                                                               self.nr_of_h_bins,
                                                               self.nr_of_w_bins,
                                                               self.network_type)
                images = gv.scale_image_by_255(images)

                self.set_parameter_dict_for_train(images, labels)

                #no = self.network_output.eval(session=self.sess, feed_dict=self.parameter_dict)
                #print(no)

                (self.train_step).run(session=self.sess, feed_dict=self.parameter_dict)

                self.set_parameter_dict_for_evaluation(images, labels)

                c_val_train = (self.C).eval(session=self.sess, feed_dict=self.parameter_dict)
                print("(in batch loop, %10s) c_val_train value: %10s" % (str(batch), str(c_val_train)))
                train_loss = train_loss + (2.0*len(mini_batch))*c_val_train

                stop = time.time()
                print("Mini batch time: " + str(stop - start))

            self.saver_loader.save(self.sess, network_model_file_name)

            print("control_step_number: " + str(self.control_step_number.eval(session=self.sess)))
            print("Evaluating the validation set.")

            glr = self.decaying_learning_rate.eval(session=self.sess)
            print("decaying_learning_rate: " + str(glr))

            gs = self.global_step.eval(session=self.sess)
            print("global_step: " + str(gs))

            train_loss = train_loss/(2.0*len(x_train))
            total_loss = self.loss_for_image_set(x_valid)

            print("Total loss: %15s Train loss: %15s" % (str(total_loss), str(train_loss)))

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

        predicted_labels = [np.zeros((self.height, self.width)) for i in range(N)]
        pl_index = 0

        mini_batch_size = 10
        n_batches = round(N / mini_batch_size)

        print("before loop")

        ptr = 0
        for n in range(n_batches):
            mini_batch = x_test[ptr:ptr + mini_batch_size]
            images, labels = gv.read_image_chunk_fish_mask(mini_batch,
                                                           self.height,
                                                           self.width,
                                                           self.nr_of_h_bins,
                                                           self.nr_of_w_bins,
                                                           self.network_type)

            ptr = ptr + mini_batch_size

            self.set_parameter_dict_for_evaluation(images, labels)

            print("in loop")
            if self.regression_network == True:
                print("regression")
                predicted_labels_mini_batch = (self.network_output_for_prediction).eval(session=self.sess,
                                                                         feed_dict=self.parameter_dict)
            elif self.classification_network == True:
                print("classification")
                predicted_labels_mini_batch = (self.network_output_for_prediction).eval(session=self.sess,
                                                                         feed_dict=self.parameter_dict)
            else:
                pass





            for j in range(len(predicted_labels_mini_batch)):
                predicted_labels[pl_index] = predicted_labels_mini_batch[j].reshape((self.nr_of_h_bins, self.nr_of_w_bins))
                pl_index = pl_index + 1


        return predicted_labels




# Some spare code

#self.network_output = tf.nn.sigmoid(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)
#self.network_output = tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected

#self.temp_const = tf.placeholder(tf.float32, shape=[None, 9 * 4], name="temp_const")
#temp = tf.nn.relu(tf.matmul(conv_layer_five_dropped_array, w_first_connected) + bias_first_connected)

#self.network_output = tf.select(tf.greater_equal(temp, self.temp_const), self.temp_const, temp)

