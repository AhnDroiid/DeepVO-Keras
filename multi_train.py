from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.optimizers import Adagrad, Adam
from keras.activations import relu, sigmoid
import glob
import cv2
import numpy as np
from model import DRVO
from util import data_generator, data_len
import config_dat as cf
import time
from Config import Vehicle

model = DRVO().model
print(model.summary())

if cf.weight_path is not None:
    model.load_weights(cf.weight_path)
    print("weight is loaded")

label_path = ["./data", "./data_1", "./data_2", "./data_3", "./data_4"]
label_name = "pose_label.txt"
for idx in range(len(label_path)):
    if idx != 0:
        model.load_weights("{}.h5".format(idx-1))
    print("{}/{} sets training start".format(idx, len(label_path)))
    train_generator = data_generator(label_path[idx], label_name)
    optimizer = Adagrad(lr=0.001)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=(data_len(label_path[idx], label_name) / cf.image_batch), epochs=50)
    model.save_weights("{}.h5".format(idx))


print("Train of {} set is finished".format(len(label_path)))


