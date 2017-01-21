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

MINI_BATCHES_FOR_LARGE_SET_PROCESSING = 250

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

    return (overlap/union)


def uoi_for_set_of_labels(labels, predicted_labels):
    N = len(labels)

    labels = [labels[i].reshape((9, 4)) for i in range(N)]
    predicted_labels = [predicted_labels[i].reshape((9, 4)) for i in range(N)]

    uoi_sum_batch = 0.0
    uoi_images_batch = 0
    for i in range(N):

        for j in range(len(labels[i])):
            x1 = labels[i][j, 0]
            y1 = labels[i][j, 1]
            w1 = labels[i][j, 2]
            h1 = labels[i][j, 3]

            """
            The uoi is calculated only for the actual fish
            images. This does not give us any information how
            well the network performs with deternining the actual
            number of fish.
            """
            if (int(x1) == -1 and
                int(y1) == -1 and
                int(w1) == -1 and
                int(h1) == -1):
                continue

            x2 = predicted_labels[i][j, 0]
            y2 = predicted_labels[i][j, 1]
            w2 = predicted_labels[i][j, 2]
            h2 = predicted_labels[i][j, 3]

            uoi_sum_batch = uoi_sum_batch + uoi(x1, y1, w1, h1, x2, y2, w2, h2)
            uoi_images_batch = uoi_images_batch + 1
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

def read_image_chunk(image_chunk_list, rfx, rfy):
    ih = int(rfy*DEFAULT_IMAGE_HEIGHT)
    iw = int(rfx*DEFAULT_IMAGE_WIDTH)

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]
    labels = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]

    for i in range(N):
        img = cv2.imread(image_chunk_list[i].file_name)
        #img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

        images[i] = img

        mask_fish = np.zeros((ih, iw))
        rects = image_chunk_list[i].rects

        for rec in rects:
            x = rec[0]
            y = rec[1]
            w = rec[2]
            h = rec[3]

            nx, ny, nw, nh = rectangle_transform(x, y, w, h, 0.0, 0.0, rfx, rfy)

            mask_fish[int(ny):int(ny+nh), int(nx):int(nx+nw)] = 1.0

        labels[i] = mask_fish

    return images, labels




def read_image_chunk_real_labels(image_chunk_list, rfx, rfy):
    #ih = int(rfy*DEFAULT_IMAGE_HEIGHT)
    #iw = int(rfx*DEFAULT_IMAGE_WIDTH)

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]
    labels = [-1.0*np.ones((9, 4)) for i in range(N)]

    #print("labels size: " + str(len(labels)))

    for i in range(N):
        #print("i: " + str(i))
        img = cv2.imread(TRAIN_FOLDER_DIR + image_chunk_list[i].file_name)
        #img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

        images[i] = img/255.0
        ih, iw, _ = images[i].shape
        rects = image_chunk_list[i].rects

        for j in range(len(rects)):
            x = rects[j][0]
            y = rects[j][1]
            w = rects[j][2]
            h = rects[j][3]

            #nx, ny, nw, nh = rectangle_transform(x, y, w, h, 0.0, 0.0, rfx, rfy)

            #print("i: " + str(i) + " j: " + str(j))
            #print("labels[i].shape: " + str(labels[i].shape))
            labels[i][j, 0] = x / iw
            labels[i][j, 1] = y / ih
            labels[i][j, 2] = w / iw
            labels[i][j, 3] = h / ih

            #mask_fish[int(ny):int(ny+nh), int(nx):int(nx+nw)] = 1.0

        #labels[i] = mask_fish
    labels = [labels[i].reshape((9 * 4, )) for i in range(N)]


    #for i in range(N):
    #    print(image_chunk_list[i].file_name)
    #    print(labels[i])


    return images, labels



def read_image_chunk_hist_labels(image_chunk_list, n_bins=256):
    #ih = int(rfy*DEFAULT_IMAGE_HEIGHT)
    #iw = int(rfx*DEFAULT_IMAGE_WIDTH)

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]

    # We have self.n_bins + 1 to mark slots that don't have rectangles.
    labels = [np.zeros((9, 4, n_bins + 1)) for i in range(N)]
    n_rects_per_img = [0 for i in range(N)]

    for i in range(N):
        img = cv2.imread(TRAIN_FOLDER_DIR + image_chunk_list[i].file_name)
        #img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

        images[i] = img/255.0
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