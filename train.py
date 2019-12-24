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

train_generator = data_generator(cf.label_path, cf.label_name)
optimizer = Adagrad(lr=0.0005)
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=(data_len(cf.label_path, cf.label_name) / cf.image_batch), epochs=100)


model.save_weights("VO_weight{}.h5".format(time.time()))
