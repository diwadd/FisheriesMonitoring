import json
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import global_variable as gv
import network_models as nm

TRAIN_FOLDER_DIR = gv.train_folder_dir
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.5
CHUNK_SIZE = 10

print("TRAIN_FOLDER_DIR %15s" % (str(VALIDATION_SIZE)))
print("TEST_SIZE %15s" % (str(TEST_SIZE)))
print("CHUNK_SIZE %15s" % (str(CHUNK_SIZE)))


def read_annotation_files(annotation_files):

    ipan = [] # image_paths_and_annotations
    for file in annotation_files:
        f = open(TRAIN_FOLDER_DIR + file, "r")
        fa = json.load(f)
        ipan = ipan + fa

    ipan = [gv.ImageAnnotation(ipan[i]) for i in range(len(ipan))]

    return ipan





annotation_files = ["y_fish_positions_ALB_1280_x_720.json",
                    "y_fish_positions_LAG_1280_x_720.json",
                    "y_fish_positions_YFT_1280_x_720.json",
                    "y_fish_positions_BET_1280_x_720.json",
                    "y_fish_positions_OTHER_1280_x_720.json",
                    "y_fish_positions_DOL_1280_x_720.json",
                    "y_fish_positions_SHARK_1280_x_720.json",
                    "y_fish_positions_Nof_1280_x_720.json",
                    "y_fish_positions_ALB_1280_x_720_cp.json",
                    "y_fish_positions_LAG_1280_x_720_cp.json",
                    "y_fish_positions_YFT_1280_x_720_cp.json",
                    "y_fish_positions_BET_1280_x_720_cp.json",
                    "y_fish_positions_OTHER_1280_x_720_cp.json",
                    "y_fish_positions_DOL_1280_x_720_cp.json",
                    "y_fish_positions_SHARK_1280_x_720_cp.json",
                    "y_fish_positions_Nof_1280_x_720_cp.json"]

ipan = read_annotation_files(annotation_files)


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

step = 30
x_train = [x_train[i] for i in range(0, len(x_train), step)]
x_valid = [x_valid[i] for i in range(0, len(x_valid), step)]
x_test = [x_test[i] for i in range(0, len(x_test), step)]

print("After reduction:")
print("x_train size: %s" % (str(len(x_train))))
print("x_valid size: %s" % (str(len(x_valid))))
print("x_test size: %s" % (str(len(x_test))))

rfx = 0.25
rfy = 0.25
print("rfx        %15s" % (str(rfx)))
print("rfy        %15s" % (str(rfy)))

#image_chunk_list = random.sample(x_train, CHUNK_SIZE)

#images, labels, _ = gv.read_image_chunk_hist_labels(image_chunk_list, rfx, rfy, n_bins=16)

#print("Histogram check")
#print(image_chunk_list[0])
#print(labels[0])

#network_type = "network_model_reg_small"
#network_type = "network_model_reg_small_xavier"
network_type = "network_model_cla_small_xavier"
channels = 3
dropout_list = [0.7, 0.7, 0.7, 0.7, 0.7]
n_bins = 256
print("network_type: " + str(network_type))
print("channels:" + str(channels))
print("dropout_list: " + str(dropout_list))
print("n_bins: " + str(n_bins))

network = nm.NeuralNetworkFishDetection(network_type, rfx, rfy, channels, dropout_list, n_bins)

n_epochs = 30
mini_batch_size = 200
print("n_epochs        %15s" % (str(n_epochs)))
print("mini_batch_size %15s" % (str(mini_batch_size)))

learning_rate = 1.0
decay_rate = 0.96
decay_steps = 18
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
              #mode="training_continuation_or_prediction",
              gv.MAIN_FOLDER_DIR + "xft_network_model_network_model_reg_small.tf",
              decay_steps,
              decay_rate)

