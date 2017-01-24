import os
import json
import shutil
import random

import cv2
import numpy as np

import global_variable as gv


random.seed(gv.RANDOM_SEED_PYTHON)
np.random.seed(gv.RANDOM_SEED_NUMPY)
train_folder_dir = gv.TRAIN_FOLDER_DIR

def get_eigen(image_directories_for_color_perturbation):

    all_data = np.array([[0,0,0]])
    for folder in image_directories_for_color_perturbation:

        os.chdir(train_folder_dir + folder)

        image_list = os.listdir()

        N = len(image_list)
        ratio = 0.1
        if folder == "ALB":
            ratio = 0.039/2
        elif folder == "BET":
            ratio = 0.33/2
        elif folder == "DOL":
            ratio = 0.52/2
        elif folder == "LAG":
            ratio = 1.0/2
        elif folder == "NoF":
            ratio = 0.14/2
        elif folder == "OTHER":
            ratio = 0.22/2
        elif folder == "SHARK":
            ratio = 0.38/2
        elif folder == "YFT":
            ratio = 0.091/2
        else:
            pass


        image_list = random.sample(image_list, int(ratio*N))

        imgs = np.empty(len(image_list), dtype=object)
        for i in range(len(image_list)):

            imgs[i] = cv2.imread(image_list[i])

        print(str(imgs[0][0, 0, 0]) + " " + str(imgs[0][1, 1, 1]))

        rows, cols, _ = imgs[0].shape
        folder_data =  np.resize(imgs[0], (rows * cols, 3))

        for i in range(1,len(imgs)):
            rows, cols, _ = imgs[i].shape
            folder_data = np.concatenate((folder_data, np.resize(imgs[i], (rows * cols, 3))), axis=0)

        all_data = np.concatenate((all_data, folder_data))

    all_data = all_data[1:]/255.0
    C = np.cov(all_data.T)
    w, v = np.linalg.eig(C)


    for folder in image_directories_for_color_perturbation:
        print(folder)
        os.chdir(train_folder_dir + folder)
        image_list = os.listdir()

        dir_with_color_perturbed_imgs = train_folder_dir + folder + "_cp/"

        if os.path.isdir(dir_with_color_perturbed_imgs) != True:
            os.mkdir(dir_with_color_perturbed_imgs)
        else:
            shutil.rmtree(dir_with_color_perturbed_imgs)
            os.mkdir(dir_with_color_perturbed_imgs)

        index = 0
        N = len(image_list)
        for image in image_list:
            print("Processing: %s" % (index/N), end="\r" )
            index = index + 1

            img = cv2.imread(image)/255.0

            rows, cols, ch = img.shape

            addition = np.dot(v, np.transpose((gv.COLOR_PERTURBATION_STD * np.random.randn(3)) * w))

            a_img = np.zeros((rows, cols, ch))
            a_img[:, :, 0] = addition[0]
            a_img[:, :, 1] = addition[1]
            a_img[:, :, 2] = addition[2]

            img = img + a_img
            img = img*255.0
            cv2.imwrite(dir_with_color_perturbed_imgs + image.replace(".jpg", "_cp.jpg"), img)
        print("Processing: %s" % (index / N), end="\r")
    print()


def resize_image(image_file_name, width, height):

    cv2_image = cv2.imread(image_file_name)
    res = cv2.resize(cv2_image, (width, height), interpolation=cv2.INTER_CUBIC)

    return res

def transform_images_rotation(image_directories_and_rotation_angles, width, height):

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

            n_angles = len(rotation_angles)
            for ang in range(n_angles):
                M = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angles[ang], 1)
                dst = cv2.warpAffine(res, M, (width, height))

                new_image_file_name = image_file_name.replace(".jpg", "_" + str(width) + "_x_" + str(height) + "_a_" + str(rotation_angles[ang]) + ".jpg")
                cv2.imwrite(directory_for_resized_images + "/" + new_image_file_name, dst)

        print("Number of files: " + str(len(os.listdir(directory_for_resized_images))))


def resize_and_shift(img, rfx, rfy, sx, sy):
    rows, cols, _ = img.shape

    res = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

    M = np.float32([[1, 0, sx], [0, 1, sy]])
    res = cv2.warpAffine(res, M, (cols, rows))

    return res


def transform_images_resize_and_shift(annotation_files, shifts_and_resize_factors):

    for af in annotation_files:
        print(af)

        f = open(train_folder_dir + af)
        annotations = json.load(f)
        new_annotations = []

        index = 0
        N = len(annotations)


        for an in annotations:
            iman = gv.ImageAnnotation(an)
            #print(iman)

            rects = iman.rects

            #print(gv.TRAIN_FOLDER_DIR + iman.file_name)
            img = cv2.imread(gv.TRAIN_FOLDER_DIR + iman.file_name)
            for srf in shifts_and_resize_factors:
                rfx = srf[0]
                rfy = srf[1]
                sx = srf[2]
                sy = srf[3]
                res = resize_and_shift(img, rfx, rfy, sx, sy)
                file_name_ext = "_rfx_" + str(rfx).replace(".","p") + "_rfy_" + str(rfy).replace(".","p") + "_sx_" + str(sx) + "_sy_" + str(sy) + ".jpg"
                new_name = iman.file_name.replace(".jpg", file_name_ext)
                cv2.imwrite(gv.TRAIN_FOLDER_DIR + new_name, res)

                annotations_list = []
                for rec in rects:
                    x0 = rec[0]
                    y0 = rec[1]
                    w0 = rec[2]
                    h0 = rec[3]

                    nx, ny, nw, nh = gv.rectangle_transform(x0, y0, w0, h0, sx, sy, rfx, rfy)

                    image_dict = {"class": "rect",
                                  "height": nh,
                                  "width": nw,
                                  "x": nx,
                                  "y": ny}
                    annotations_list.append(image_dict)

                annotations_dict = {"annotations": annotations_list,
                                    "class": "image",
                                    "filename": new_name}
                new_annotations.append(annotations_dict)
            index = index + 1
            print("Processed: %10s" % (str(index/N)), end="\r")
        print()


        f.close()

        new_file_name = train_folder_dir + af
        print(new_file_name)

        g = open(new_file_name, "w+")
        json_file_content = json.dump(annotations + new_annotations, g, sort_keys=True, indent=4, ensure_ascii=False)
        g.close()
        print()


def parse_json(annotation_files, width, height, ratio):

    new_annotation_files = []
    for af in annotation_files:

        f = open(train_folder_dir + af)
        annotations = json.load(f)
        new_annotations = []

        for an in annotations:

            iman = gv.ImageAnnotation(an)
            rects = iman.rects
            annotations_list = []
            for rec in rects:
                x0 = rec[0]
                y0 = rec[1]
                w0 = rec[2]
                h0 = rec[3]

                nx, ny, nw, nh = gv.rectangle_transform(x0, y0, w0, h0, 0.0, 0.0, ratio, ratio)

                image_dict = {"class": "rect",
                              "height": nh,
                              "width": nw,
                              "x": nx,
                              "y": ny}
                annotations_list.append(image_dict)

            new_name = iman.file_name
            new_name = new_name.replace("1280_x_720", str(width) + "_x_" + str(height))

            annotations_dict = {"annotations": annotations_list,
                                "class": "image",
                                "filename": new_name}
            new_annotations.append(annotations_dict)

        new_annotation_file_name = "y_" + af.replace("1280_x_720", str(width) + "_x_" + str(height))
        g = open(gv.TRAIN_FOLDER_DIR + new_annotation_file_name, "w")
        json_file_content = json.dump(new_annotations, g, sort_keys=True, indent=4, ensure_ascii=False)
        g.close()
        new_annotation_files.append(new_annotation_file_name)
    return new_annotation_files


image_directories_for_color_perturbation = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]


get_eigen(image_directories_for_color_perturbation)


image_directories_and_rotation_angles = {"ALB": [0],
                                         "ALB_cp": [0],
                                         "BET": [-5, -3, -1, 0, 2, 4, 6, 8],
                                         "BET_cp": [-5, -3, -1, 0, 2, 4, 6, 8],
                                         "DOL": [-13, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 10, 12],
                                         "DOL_cp": [-13, -11, -9, -7, -5, -3, -1, 0, 2, 4, 6, 8, 10, 12],
                                         "LAG": [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                         "LAG_cp": [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                         "NoF": [-5, 0, 2.5, 5],
                                         "NoF_cp": [-5, 0, 2.5, 5],
                                         "OTHER": [-4, -2, 0, 3, 5, 7],
                                         "OTHER_cp": [-4, -2, 0, 3, 5, 7],
                                         "SHARK": [-8, -6, -4, -2, 0, 1, 3, 5, 7, 9],
                                         "SHARK_cp": [-8, -6, -4, -2, 0, 1, 3, 5, 7, 9],
                                         "YFT": [-5, 0],
                                         "YFT_cp": [-5, 0]}

ratio = 0.4
default_width = 1280
default_height = 720

transform_images_rotation(image_directories_and_rotation_angles, int(ratio*default_width), int(ratio*default_height))


annotation_files = ["ind_fish_positions_ALB_1280_x_720.json",
                    "ind_fish_positions_LAG_1280_x_720.json",
                    "ind_fish_positions_YFT_1280_x_720.json",
                    "ind_fish_positions_BET_1280_x_720.json",
                    "ind_fish_positions_OTHER_1280_x_720.json",
                    "ind_fish_positions_DOL_1280_x_720.json",
                    "ind_fish_positions_SHARK_1280_x_720.json",
                    "ind_fish_positions_Nof_1280_x_720.json",
                    "ind_fish_positions_ALB_1280_x_720_cp.json",
                    "ind_fish_positions_LAG_1280_x_720_cp.json",
                    "ind_fish_positions_YFT_1280_x_720_cp.json",
                    "ind_fish_positions_BET_1280_x_720_cp.json",
                    "ind_fish_positions_OTHER_1280_x_720_cp.json",
                    "ind_fish_positions_DOL_1280_x_720_cp.json",
                    "ind_fish_positions_SHARK_1280_x_720_cp.json",
                    "ind_fish_positions_Nof_1280_x_720_cp.json"]


new_annotation_files = parse_json(annotation_files,
                                  int(ratio*default_width),
                                  int(ratio*default_height),
                                  ratio)

shifts_and_resize_factors = [[0.95, 0.95, 0, 0],
                             [0.95, 0.95, 15, 0],
                             [0.95, 0.95, 15, 15],
                             [0.95, 0.95, 0, 15]]


transform_images_resize_and_shift(new_annotation_files, shifts_and_resize_factors)
