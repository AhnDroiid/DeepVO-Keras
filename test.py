import numpy as np
from model import DRVO
from util import data_generator, data_len, sample_data_batch
import config_dat as cf
from keras.optimizers import Adagrad, Adam
model = DRVO().model
print(model.summary())

optimizer = Adam(lr=0.0005)
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['accuracy'])

if cf.weight_path is not None:
    model.load_weights(cf.weight_path)
    print("weight is loaded")

test_generator = data_generator(cf.label_path, cf.label_name)
x_true, y_true, image_name = sample_data_batch(cf.label_path, cf.label_name, cf.image_batch)

###### Prediction with multiple batch ######

# result_list = []
# result = (model.predict_generator(test_generator, steps=data_len() / cf.image_batch))


###### Prediction with single batch #####
print(x_true.shape)
result = model.evaluate(x_true, y_true, batch_size=cf.image_batch)
prediction = model.predict(x_true, batch_size=cf.image_batch)
print(result)
print(prediction)
print(y_true)
#print(image_name)

