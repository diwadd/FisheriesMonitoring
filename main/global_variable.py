import random
import cv2
import numpy as np

EPSILON_BN = 1e-3
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720

COLOR_PERTURBATION_STD = 0.1

train_folder_dir = "/home/tadek/Coding_Competitions/Kaggle/FisheriesMonitoring/train/"

INT32_MAX = 2147483647

RANDOM_SEED_PYTHON = random.randrange(1, INT32_MAX - 1)
RANDOM_SEED_NUMPY = random.randrange(1, INT32_MAX - 1)

SK_LEARN_RANDOM_STATE = random.randrange(1, INT32_MAX - 1)
TENSORFLOW_RANDOM_STATE = random.randrange(1, INT32_MAX - 1)

print("RANDOM_SEED_PYTHON      %20s" % (str(RANDOM_SEED_PYTHON)))
print("RANDOM_SEED_NUMPY       %20s" % (str(RANDOM_SEED_NUMPY)))
print("SK_LEARN_RANDOM_STATE   %20s" % (str(SK_LEARN_RANDOM_STATE)))
print("TENSORFLOW_RANDOM_STATE %20s" % (str(TENSORFLOW_RANDOM_STATE)))


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


def rectangle_transform(x, y, w, h, sx, sy, rfx, rfy):
    """

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
        img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

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
    ih = int(rfy*DEFAULT_IMAGE_HEIGHT)
    iw = int(rfx*DEFAULT_IMAGE_WIDTH)

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]
    labels = [np.zeros((9, 4)) for i in range(N)]

    #print("labels size: " + str(len(labels)))

    for i in range(N):
        #print("i: " + str(i))
        img = cv2.imread(image_chunk_list[i].file_name)
        img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

        images[i] = img/255.0
        rects = image_chunk_list[i].rects
        #print("rects size: " + str(len(rects)))

        for j in range(len(rects)):
            x = rects[j][0]
            y = rects[j][1]
            w = rects[j][2]
            h = rects[j][3]

            nx, ny, nw, nh = rectangle_transform(x, y, w, h, 0.0, 0.0, rfx, rfy)

            #print("i: " + str(i) + " j: " + str(j))
            #print("labels[i].shape: " + str(labels[i].shape))
            labels[i][j, 0] = nx/iw
            labels[i][j, 1] = ny/ih
            labels[i][j, 2] = nw/iw
            labels[i][j, 3] = nh/ih

            #mask_fish[int(ny):int(ny+nh), int(nx):int(nx+nw)] = 1.0

        #labels[i] = mask_fish
    labels = [labels[i].reshape((9 * 4, )) for i in range(N)]


    #for i in range(N):
    #    print(image_chunk_list[i].file_name)
    #    print(labels[i])


    return images, labels



def read_image_chunk_hist_labels(image_chunk_list, rfx, rfy, n_bins=256):
    ih = int(rfy*DEFAULT_IMAGE_HEIGHT)
    iw = int(rfx*DEFAULT_IMAGE_WIDTH)

    N = len(image_chunk_list)

    images = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]
    labels = [np.zeros((DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)) for i in range(N)]

    for i in range(N):
        img = cv2.imread(image_chunk_list[i].file_name)
        img = cv2.resize(img, None, fx=rfx, fy=rfy, interpolation=cv2.INTER_CUBIC)

        images[i] = img/255.0

        mask_fish = np.zeros((ih, iw))
        rects = image_chunk_list[i].rects

        labels = np.zeros((9,n_bins))

        for i in range(len(rects)):
            x = rects[i][0]
            y = rects[i][1]
            w = rects[i][2]
            h = rects[i][3]

            nx, ny, nw, nh = rectangle_transform(x, y, w, h, 0.0, 0.0, rfx, rfy)

            x_bin_id = round(nx / n_bins)
            y_bin_id = round(ny / n_bins)
            w_bin_id = round(nw / n_bins)
            h_bin_id = round(nh / n_bins)

            labels[i][x_bin_id] = 1.0
            labels[i][y_bin_id] = 1.0
            labels[i][w_bin_id] = 1.0
            labels[i][h_bin_id] = 1.0

            #mask_fish[int(ny):int(ny+nh), int(nx):int(nx+nw)] = 1.0

        labels = labels/(4*len(rects))
        labels = np.resize(labels, (1, 9 * n_bins))

    return images, labels