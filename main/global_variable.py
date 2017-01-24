import random
import sys
import datetime
import json


import cv2
import numpy as np

EPSILON_BN = 1e-3
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720

COLOR_PERTURBATION_STD = 0.1

TRAIN_FOLDER_DIR = "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/"

MAIN_FOLDER_DIR = "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/main/"

MINI_BATCHES_FOR_LARGE_SET_PROCESSING = 600

#CV_READ_OPTION = 1
CV_READ_OPTION = 0

UINT32_MAX = 4294967295

RANDOM_SEED_PYTHON = random.randrange(1, UINT32_MAX - 1)
RANDOM_SEED_NUMPY = random.randrange(1, UINT32_MAX - 1)

SK_LEARN_RANDOM_STATE = random.randrange(1, UINT32_MAX - 1)
TENSORFLOW_RANDOM_STATE = random.randrange(1, UINT32_MAX - 1)

print("EPSILON_BN                             %20s" % (str(EPSILON_BN)))
print("DEFAULT_IMAGE_WIDTH                    %20s" % (str(DEFAULT_IMAGE_WIDTH)))
print("DEFAULT_IMAGE_HEIGHT                   %20s" % (str(DEFAULT_IMAGE_HEIGHT)))

print("COLOR_PERTURBATION_STD                 %20s" % (str(COLOR_PERTURBATION_STD)))

print("MINI_BATCHES_FOR_LARGE_SET_PROCESSING  %20s" % (str(MINI_BATCHES_FOR_LARGE_SET_PROCESSING)))

print("CV_READ_OPTION:                        %20s" % (str(CV_READ_OPTION)))

print("RANDOM_SEED_PYTHON                     %20s" % (str(RANDOM_SEED_PYTHON)))
print("RANDOM_SEED_NUMPY                      %20s" % (str(RANDOM_SEED_NUMPY)))
print("SK_LEARN_RANDOM_STATE                  %20s" % (str(SK_LEARN_RANDOM_STATE)))
print("TENSORFLOW_RANDOM_STATE                %20s" % (str(TENSORFLOW_RANDOM_STATE)))


class ImageAnnotation:

    def prase_annotations(self, annotations):
        N = len(annotations)

        rects = [[0, 0, 0, 0] for i in range(N)]
        for i in range(N):
            rects[i][0] = annotations[i]["x"]
            rects[i][1] = annotations[i]["y"]
            rects[i][2] = annotations[i]["width"]
            rects[i][3] = annotations[i]["height"]

        return rects

    def __init__(self, image_annotations):
        self.file_name = image_annotations["filename"]
        self.rects = self.prase_annotations(image_annotations["annotations"])

    def __str__(self):
        s = "Filename: " + str(self.file_name) + "\n"
        s = s + ("Number of rectangles: " + str(len(self.rects))) + "\n"
        for i in range(len(self.rects)):
            x = self.rects[i][0]
            y = self.rects[i][1]
            w = self.rects[i][2]
            h = self.rects[i][3]
            s = s + "x: %6s y: %6s w: %6s h: %6s\n" % (x, y, w, h)
        return s


def read_annotation_files(annotation_files):

    ipan = [] # image_paths_and_annotations
    for file in annotation_files:
        f = open(TRAIN_FOLDER_DIR + file, "r")
        fa = json.load(f)
        ipan = ipan + fa

    ipan = [ImageAnnotation(ipan[i]) for i in range(len(ipan))]

    return ipan


def uoi(x1, y1, w1, h1, x2, y2, w2, h2):

    ax1 = x1
    ax2 = x1 + w1
    ay1 = y1
    ay2 = y1 + h1

    bx1 = x2
    bx2 = x2 + w2
    by1 = y2
    by2 = y2 + h2

    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area1 = abs(ax1 - ax2) * abs(ay1 - ay2)
    area2 = abs(bx1 - bx2) * abs(by1 - by2)

    union = area1 + area2 - overlap
    #print("union: " + str(union))

    if round(union, 9) == 0.0:
        return 0.0
    else:
        return (overlap/union)


def uoi_for_set_of_labels(labels,
                          predicted_labels,
                          n_rects_per_img):
    N = len(labels)

    labels = [labels[i].reshape((9, 4)) for i in range(N)]
    predicted_labels = [predicted_labels[i].reshape((9, 4)) for i in range(N)]

    uoi_sum_batch = 0.0
    uoi_images_batch = 0
    for i in range(N):

        M = n_rects_per_img[i]

        for j in range(M):
            x1 = labels[i][j, 0]
            y1 = labels[i][j, 1]
            w1 = labels[i][j, 2]
            h1 = labels[i][j, 3]

            #print("x1: %8s y1: %8s w1: %8s h1: %8s" % (str(x1), str(y1), str(w1), str(h1)))

            """
            The uoi is calculated only for the actual fish
            images. This does not give us any information how
            well the network performs with deternining the actual
            number of fish.
            """
            #if (round(x1, 9) == 0.0 and
            #    round(y1, 9) == 0.0 and
            #    round(w1, 9) == 0.0 and
            #    round(h1, 9) == 0.0):
            #    continue

            x2 = predicted_labels[i][j, 0]
            y2 = predicted_labels[i][j, 1]
            w2 = predicted_labels[i][j, 2]
            h2 = predicted_labels[i][j, 3]

            #print("x2: %8s y2: %8s w2: %8s h2: %8s" % (str(x2), str(y2), str(w2), str(h2)))

            uoi_sum_batch = uoi_sum_batch + uoi(x1, y1, w1, h1, x2, y2, w2, h2)
            #print("uoi_sum_batch: " + str(uoi_sum_batch))
            uoi_images_batch = uoi_images_batch + 1
    #if uoi_images_batch == 0:
    #    sys.exit("ERROR! uoi_images_batch == 0")
    return uoi_sum_batch, uoi_images_batch



def uoi_for_set_of_labels_cla_version(labels,
                                      predicted_labels,
                                      binning_array_width,
                                      binning_array_height,
                                      n_rects_per_img):
    N = len(labels)

    n_bins = len(binning_array_width)

    labels = [labels[i].reshape((9, 4, n_bins + 1)) for i in range(N)]
    predicted_labels = [predicted_labels[i].reshape((9, 4, n_bins + 1)) for i in range(N)]

    uoi_sum_batch = 0.0
    uoi_images_batch = 0
    for i in range(N):

        M = n_rects_per_img[i]
        """
        The uoi is calculated only for the actual fish
        images. This does not give us any information how
        well the network performs with deternining the actual
        number of fish.
        """
        for j in range(M):

            x1_index = np.argmax(labels[i][j, 0, :-1])
            y1_index = np.argmax(labels[i][j, 1, :-1])
            w1_index = np.argmax(labels[i][j, 2, :-1])
            h1_index = np.argmax(labels[i][j, 3, :-1])

            x1 = binning_array_width[x1_index]
            y1 = binning_array_height[y1_index]
            w1 = binning_array_width[w1_index]
            h1 = binning_array_height[h1_index]

            x2_index = np.argmax(predicted_labels[i][j, 0, :-1])
            y2_index = np.argmax(predicted_labels[i][j, 1, :-1])
            w2_index = np.argmax(predicted_labels[i][j, 2, :-1])
            h2_index = np.argmax(predicted_labels[i][j, 3, :-1])

            x2 = binning_array_width[x2_index]
            y2 = binning_array_height[y2_index]
            w2 = binning_array_width[w2_index]
            h2 = binning_array_height[h2_index]

            uoi_sum_batch = uoi_sum_batch + uoi(x1, y1, w1, h1, x2, y2, w2, h2)
            uoi_images_batch = uoi_images_batch + 1
    return uoi_sum_batch, uoi_images_batch



def rectangle_transform(x, y, w, h, sx, sy, rfx, rfy):
    """
    Scale the rectangle that marks a fish by rfx in the x direction and by
    rfy in the y direction. Shifts that rectangle by sx and by sy.

    :param x:
    :param y:
    :param w:
    :param h:
    :param sx:
    :param sy:
    :param rfx:
    :param rfy:
    :return:
    """

    nx = int(rfx*x) + sx
    ny = int(rfy*y) + sy

    bx = int(rfx*(x+w)) + sx
    by = int(rfy*(y+h)) + sy

    nw = abs(nx - bx)
    nh = abs(ny - by)

    return nx, ny, nw, nh


def resize_fish_mask_to_original_image_size(fish_mask, height, width, nr_of_h_bins, nr_of_w_bins):

    rfx = nr_of_w_bins/width
    rfy = nr_of_h_bins/height

    resized_fish_mask = cv2.resize(fish_mask,
                                   None,
                                   fx=(1.0 / rfx),
                                   fy=(1.0 / rfy),
                                   interpolation=cv2.INTER_CUBIC)

    resized_fish_mask[resized_fish_mask != 0] = 1.0

    return resized_fish_mask



def read_image_chunk_fish_mask(image_chunk_list,
                               height,
                               width,
                               nr_of_h_bins,
                               nr_of_w_bins,
                               network_type):

    N = len(image_chunk_list)

    images = [np.zeros((height, width)) for i in range(N)]
    labels = [np.zeros((nr_of_h_bins, nr_of_w_bins)) for i in range(N)]

    for i in range(N):
        img = cv2.imread(TRAIN_FOLDER_DIR + image_chunk_list[i].file_name, CV_READ_OPTION)

        if CV_READ_OPTION == 0:
            img = img.reshape((height, width, 1))

        images[i] = img

        fish_mask = np.zeros((nr_of_h_bins, nr_of_w_bins))
        rects = image_chunk_list[i].rects


        for rec in rects:
            ix = rec[0]
            iy = rec[1]
            iw = rec[2]
            ih = rec[3]

            x1 = ix
            y1 = iy
            x2 = ix + iw
            y2 = iy + ih

            x1_bin = int((x1 / width) * nr_of_w_bins)
            x2_bin = int((x2 / width) * nr_of_w_bins)

            y1_bin = int((y1 / height) * nr_of_h_bins)
            y2_bin = int((y2 / height) * nr_of_h_bins)

            fish_mask[y1_bin:y2_bin, x1_bin:x2_bin] = 1.0


        if network_type == "classification":
            s = np.sum(fish_mask)
            if int(s) != 0:
                fish_mask = fish_mask/s

        labels[i] = fish_mask

    labels = [labels[i].reshape((nr_of_h_bins*nr_of_w_bins, )) for i in range(N)]

    return images, labels




def read_image_chunk_real_labels(image_chunk_list):

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]
    labels = [np.zeros((9, 4)) for i in range(N)]
    n_rects_per_img = [0 for i in range(N)]

    for i in range(N):
        img = cv2.imread(TRAIN_FOLDER_DIR + image_chunk_list[i].file_name, CV_READ_OPTION)


        images[i] = img
        ih, iw, _ = images[i].shape
        rects = image_chunk_list[i].rects

        M = len(rects)
        n_rects_per_img[i] = M

        for j in range(M):
            x = rects[j][0]
            y = rects[j][1]
            w = rects[j][2]
            h = rects[j][3]

            labels[i][j, 0] = x / iw
            labels[i][j, 1] = y / ih
            labels[i][j, 2] = w / iw
            labels[i][j, 3] = h / ih

    labels = [labels[i].reshape((9 * 4, )) for i in range(N)]

    return images, labels, n_rects_per_img



def read_image_chunk_hist_labels(image_chunk_list, n_bins=256):
    #ih = int(rfy*DEFAULT_IMAGE_HEIGHT)
    #iw = int(rfx*DEFAULT_IMAGE_WIDTH)

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]

    # We have self.n_bins + 1 to mark slots that don't have rectangles.
    labels = [np.zeros((9, 4, n_bins + 1)) for i in range(N)]
    n_rects_per_img = [0 for i in range(N)]

    for i in range(N):
        img = cv2.imread(TRAIN_FOLDER_DIR + image_chunk_list[i].file_name, CV_READ_OPTION)
        #img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)


        images[i] = img
        ih, iw, _ = images[i].shape
        rects = image_chunk_list[i].rects

        M = len(rects)
        n_rects_per_img[i] = M

        for j in range(M):
            x = rects[j][0]
            y = rects[j][1]
            w = rects[j][2]
            h = rects[j][3]

            x_bin_id = int(x/iw * n_bins)
            y_bin_id = int(y/ih * n_bins)
            w_bin_id = int(w/iw * n_bins)
            h_bin_id = int(h/ih * n_bins)

            labels[i][j, 0, x_bin_id] = 1.0
            labels[i][j, 1, y_bin_id] = 1.0
            labels[i][j, 2, w_bin_id] = 1.0
            labels[i][j, 3, h_bin_id] = 1.0


        # Mark the last bins to 1.0 since there are no rectangles left.
        labels[i][M:, :, n_bins] = 1.0
        norm = np.sum(labels[i])

        labels[i] = labels[i]/norm

    labels = [labels[i].reshape((9 * 4 * (n_bins + 1))) for i in range(N)]

    return images, labels, n_rects_per_img



def add_sobel_edge(img):

    h, w, c = img.shape
    new_img = np.zeros((h, w, c + 1))

    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    new_img[:, :, 0:c] = img

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    new_img[:, :, c] = sobelx + sobely

    return new_img


def add_sobel_edge_on_images(images):
    N = len(images)
    for i in range(N):
        print("Processing: %10s" % (str((i+1)/N)), end = "\r")
        images[i] = add_sobel_edge(images[i])
    print("Exting add_sobel_edge_on_images")
    return images



def scale_image_by_255(images):
    #N = len(images)
    #for i in range(N):
    #    images[i] = images[i]/255.0

    return [images[i]/255.0 for i in range(len(images))]







