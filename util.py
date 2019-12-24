import cv2
import numpy as np
import config_dat as cf
import os
from Config import Vehicle
import matplotlib.pyplot as plt
from time import sleep

def harris(file):
    img = cv2.imread(file)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.1)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img


def orb(file):
    img = cv2.imread(file, 0)

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    #kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.detectAndCompute(img, None)

    print(kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None,  color=(0, 255, 0), flags=0)

    return img2

def data_len(label_path, label_name):
    label_path = os.path.join(label_path, label_name)
    fp = open(label_path, 'r')
    lines = fp.readlines()
    line_list = []
    for line in lines:
        line_list.append(line.strip().split(" "))
    fp.close()
    data_len = len(line_list)
    return data_len

def sample_data_batch(label_path, label_name, size):
    label_path = os.path.join(label_path, label_name)
    fp = open(label_path, 'r')
    lines = fp.readlines()
    line_list = []
    # print(lines)
    for idx, line in enumerate(lines):
        line_list.append(line.strip().split(" "))
    # print(line_list)
    fp.close()

    data_len = len(line_list)
    image_stack = cf.image_stack

    x_true = []
    y_true = []
    image_name = []
    for batch in range(size):
        start_point = int(np.random.choice(range(1, (data_len - 1 - image_stack)), 1))
        x_stack = []
        y_stack = []
        image_name.append(line_list[start_point][0])
        for indice in range(start_point, start_point + image_stack):
            train_img = cv2.imread(line_list[indice][0]).astype(np.float32) / 255
            train_img -= train_img.mean()
            train_img /= train_img.std()

            x_stack.append(train_img)
            y_stack.append(line_list[indice][7:10])
        x_true.append(np.asarray(x_stack))
        y_true.append(np.asarray(y_stack))

    x_true = np.asarray(x_true)
    y_true = np.asarray(y_true)
    y_true = y_true[:, 1:, :]

    return x_true, y_true, image_name


def data_generator(label_path, label_name):
    batch_size = cf.image_batch
    image_stack = cf.image_stack
    label_path = os.path.join(label_path, label_name)
    fp = open(label_path, 'r')
    lines = fp.readlines()
    line_list = []
    for idx, line in enumerate(lines):
        line_list.append(line.strip().split(" "))
    fp.close()
    data_len = len(line_list)


    while True:
        x_true = []
        y_true = []
        for batch in range(batch_size):
            start_point = int(np.random.choice(range(1, (data_len - 1 - image_stack)), 1))
            x_stack = []
            y_stack = []
            for indice in range(start_point, start_point + image_stack):
                train_img = cv2.imread(line_list[indice][0]).astype(np.float32) / 255
                train_img -= train_img.mean()
                train_img /= train_img.std()

                x_stack.append(train_img)
                y_stack.append(line_list[indice][7:10])


            x_true.append(np.asarray(x_stack))
            y_true.append(np.asarray(y_stack))

        x_true = np.asarray(x_true)
        y_true = np.asarray(y_true)
        y_true = y_true[:, 1:, :]
        #print(y_true)

        yield x_true, y_true


# x_true , y_true = (data_generator())
#
# x_true = np.asarray(x_true)
# y_true = np.asarray(y_true)
#
# print(y_true.shape)

# x_true, y_true, image_name = sample_data_one_batch()
# print(image_name)
# print(x_true)
# print(y_true)