import os
import cv2
import shutil

train_folder_dir = "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/"

def resize_image(image_file_name, width, height):

    cv2_image = cv2.imread(image_file_name)
    res = cv2.resize(cv2_image, (width, height), interpolation=cv2.INTER_CUBIC)

    return res

def transform_images(image_directories, width, height):

    n_folders = len(image_directories)
    for i in range(n_folders):

        directory_for_resized_images = train_folder_dir + image_directories[i] + "_" + str(width) + "_x_" + str(height)
        if os.path.isdir(directory_for_resized_images) != True:
            os.mkdir(directory_for_resized_images)
        else:
            shutil.rmtree(directory_for_resized_images)
            os.mkdir(directory_for_resized_images)

        print("We are at: " + image_directories[i])
        os.chdir(train_folder_dir + image_directories[i])

        image_list = os.listdir()
        n_files = len(image_list)
        for j in range(n_files):
            image_file_name = image_list[j]
            res = resize_image(image_file_name, width, height)

            new_image_file_name = image_file_name.replace(".jpg", "_" + str(width) + "_x_" + str(height) + ".jpg")
            cv2.imwrite(directory_for_resized_images + "/" + new_image_file_name, res)


image_directories = [
    "ALB",
    "BET",
    "DOL",
    "LAG",
    "NoF",
    "OTHER",
    "SHARK",
    "YFT"
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