import json
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import global_variable as gv
import network_models as nm

VALIDATION_SIZE = 0.2
TEST_SIZE = 0.5
CHUNK_SIZE = 10

print("VALIDATION_SIZE %15s" % (str(VALIDATION_SIZE)))
print("TEST_SIZE %15s" % (str(TEST_SIZE)))
print("CHUNK_SIZE %15s" % (str(CHUNK_SIZE)))


annotation_files = ["y_ind_fish_positions_ALB_320_x_180.json",
                    "y_ind_fish_positions_LAG_320_x_180.json",
                    "y_ind_fish_positions_YFT_320_x_180.json",
                    "y_ind_fish_positions_BET_320_x_180.json",
                    "y_ind_fish_positions_OTHER_320_x_180.json",
                    "y_ind_fish_positions_DOL_320_x_180.json",
                    "y_ind_fish_positions_SHARK_320_x_180.json",
                    "y_ind_fish_positions_Nof_320_x_180.json",
                    "y_ind_fish_positions_ALB_320_x_180_cp.json",
                    "y_ind_fish_positions_LAG_320_x_180_cp.json",
                    "y_ind_fish_positions_YFT_320_x_180_cp.json",
                    "y_ind_fish_positions_BET_320_x_180_cp.json",
                    "y_ind_fish_positions_OTHER_320_x_180_cp.json",
                    "y_ind_fish_positions_DOL_320_x_180_cp.json",
                    "y_ind_fish_positions_SHARK_320_x_180_cp.json",
                    "y_ind_fish_positions_Nof_320_x_180_cp.json"]

print(annotation_files)

ipan = gv.read_annotation_files(annotation_files)


print("Total number of images available for training: " + str(len(ipan)))


dummy_labels = [0 for i in range(len(ipan))]
x_train, x_valid, _, _ = train_test_split(ipan,
                                          dummy_labels,
                                          test_size=VALIDATION_SIZE,
                                          random_state=gv.SK_LEARN_RANDOM_STATE)

dummy_labels = [0 for i in range(len(x_valid))]
x_valid, x_test, _, _ = train_test_split(x_valid,
                                        dummy_labels,
                                        test_size=TEST_SIZE,
                                        random_state=(gv.SK_LEARN_RANDOM_STATE+2))


print("Number of images: %s" % (str(len(ipan))))
print("x_train size: %s" % (str(len(x_train))))
print("x_valid size: %s" % (str(len(x_valid))))
print("x_test size: %s" % (str(len(x_test))))

step = 1
x_train = [x_train[i] for i in range(0, len(x_train), step)]
x_valid = [x_valid[i] for i in range(0, len(x_valid), step)]
x_test = [x_test[i] for i in range(0, len(x_test), step)]

print("After reduction:")
print("x_train size: %s" % (str(len(x_train))))
print("x_valid size: %s" % (str(len(x_valid))))
print("x_test size: %s" % (str(len(x_test))))

#rfx = 0.25
#rfy = 0.25
#print("rfx        %15s" % (str(rfx)))
#print("rfy        %15s" % (str(rfy)))

#image_chunk_list = random.sample(x_train, CHUNK_SIZE)

#images, labels, _ = gv.read_image_chunk_hist_labels(image_chunk_list, n_bins=16)

#print("Histogram check")
#print(image_chunk_list[0])
#print(labels[0].reshape(9,4,17))

#network_type = "network_model_reg_small"
#network_type = "network_model_reg_small_xavier"
network_type = "network_model_cla_micro_xavier"
#network_type = "network_model_cla_small_xavier"
#network_type = "network_model_cla_small_xavier_ver2"

channels = 3
dropout_list = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
n_bins = 128
width = 320
height = 180
print("network_type: " + str(network_type))
print("channels:" + str(channels))
print("dropout_list: " + str(dropout_list))
print("n_bins: " + str(n_bins))

network = nm.NeuralNetworkFishDetection(network_type,
                                        width, height,
                                        channels,
                                        dropout_list,
                                        n_bins)

n_epochs = 10000
mini_batch_size = 400
print("n_epochs        %15s" % (str(n_epochs)))
print("mini_batch_size %15s" % (str(mini_batch_size)))

learning_rate = 10.0
decay_rate = 0.6
decay_steps = 270

print("learning_rate   %15s" % (str(learning_rate)))
print("decay_rate: %10s" % (str(decay_rate)))
print("decay_steps: %10s" % (str(decay_steps)))

#predicted_labels = network.predict(x_test, network_model_file_name=gv.MAIN_FOLDER_DIR + "xft_network_model_network_model_reg_small.tf")

#print(predicted_labels)

network.train(x_train,
              x_valid,
              n_epochs,
              mini_batch_size,
              learning_rate,
              "training",
              #"training_continuation_or_prediction",
              gv.MAIN_FOLDER_DIR + "xft_network_model_network_model_reg_small.tf",
              decay_steps,
              decay_rate)

