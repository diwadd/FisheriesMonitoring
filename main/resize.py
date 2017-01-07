import os
import cv2
import shutil

train_folder_dir = "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/"

def resize_image(image_file_name, width, height):

    cv2_image = cv2.imread(image_file_name)
    res = cv2.resize(cv2_image, (width, height), interpolation=cv2.INTER_CUBIC)

    return res

def transform_images(image_directories_and_rotation_angles, width, height):

    #n_folders = len(image_directories_and_rotation_angles)
    for folder, rotation_angles in image_directories_and_rotation_angles.items():

        directory_for_resized_images = train_folder_dir + folder + "_" + str(width) + "_x_" + str(height)
        if os.path.isdir(directory_for_resized_images) != True:
            os.mkdir(directory_for_resized_images)
        else:
            shutil.rmtree(directory_for_resized_images)
            os.mkdir(directory_for_resized_images)

        print("We are at: " + folder)
        os.chdir(train_folder_dir + folder)

        image_list = os.listdir()
        n_files = len(image_list)
        for j in range(n_files):
            image_file_name = image_list[j]
            res = resize_image(image_file_name, width, height)
            #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            n_angles = len(rotation_angles)
            for ang in range(n_angles):
                M = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angles[ang], 1)
                dst = cv2.warpAffine(res, M, (width, height))

                new_image_file_name = image_file_name.replace(".jpg", "_" + str(width) + "_x_" + str(height) + "_a_" + str(rotation_angles[ang]) + ".jpg")
                cv2.imwrite(directory_for_resized_images + "/" + new_image_file_name, dst)

        print("Number of files: " + str(len(os.listdir(directory_for_resized_images))))

"""
image_directories_and_rotation_angles = ["ALB",
    "BET",
    "DOL",
    "LAG",
    "NoF",
    "OTHER",
    "SHARK",
    "YFT"
    ]
"""

image_directories_and_rotation_angles = {"ALB": [0],
                                         "BET": [-5, -3, -1, 0, 2, 4, 6, 8],
                                         "DOL": [-13, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 10, 12],
                                         "LAG": [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                         "NoF": [-5, 0, 2.5, 5],
                                         "OTHER": [-4, -2, 0, 3, 5, 7],
                                         "SHARK": [-8, -6, -4, -2, 0, 1, 3, 5, 7, 9],
                                         "YFT": [-5, 0]}

ratio = 1.0
default_width = 1280
default_height = 720

transform_images(image_directories_and_rotation_angles, int(ratio*default_width), int(ratio*default_height))


