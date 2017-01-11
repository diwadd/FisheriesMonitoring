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


print("number of images: " + str(len(ipan)))

print(ipan[0])

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


image_chunk_list = random.sample(x_train, CHUNK_SIZE)

rfx = 0.25
rfy = 0.25

images, labels = gv.read_image_chunk(image_chunk_list, rfx, rfy)

print(images[0].shape)
print(labels[0].shape)

network = nm.NeuralNetworkFishDetection("network_model_reg_small", rfx, rfy, 3, [0.7, 0.7, 0.7, 0.7, 0.7])
#network = nm.NeuralNetworkFishDetection("network_model_reg_small_xavier", rfx, rfy, 3, [0.7, 0.7, 0.7, 0.7, 0.7])

n_epochs = 10
mini_batch_size = 200
learning_rate = 1.0

network.train(x_train,
              x_valid,
              n_epochs,
              mini_batch_size,
              learning_rate,
              "network_model_network_model_reg_small.tf")