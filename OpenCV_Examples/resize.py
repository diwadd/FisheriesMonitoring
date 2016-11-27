import os
import cv2


def resize_image(image_file_name, width, height):

    cv2_image = cv2.imread(image_file_name)
    res = cv2.resize(cv2_image, (width, height), interpolation=cv2.INTER_CUBIC)

    new_image_file_name = image_file_name.replace(".jpg", "_"+ str(width) + "_x_" + str(height) + ".jpg")
    cv2.imwrite(new_image_file_name, res)


def transform_images(image_directories, width, height):

    n_folders = len(image_directories)
    for i in range(n_folders):
        print("We are at: " + image_directories[i])
        os.chdir(image_directories[i])
        image_list = os.listdir()

        n_files = len(image_list)
        for j in range(n_files):
            resize_image(image_list[j], width, height)


image_directories = [
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/ALB",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/BET",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/DOL",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/LAG",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/NoF",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/OTHER",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/SHARK",
    "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/YFT"
    ]


ratio = 1.0
default_width = 1280
default_height = 720

transform_images(image_directories, int(ratio*default_width), int(ratio*default_height))

#resize_image("img_00003.jpg", int(ratio*default_width), int(ratio*default_height))

# Change directory
#os.chdir(alb_directory)

# Check directory
#s_pwd = os.getcwd()
#print("Current directory: " + str(s_pwd))

# Get file list
#image_files = os.listdir()
#print(image_files)