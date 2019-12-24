import time

image_width = 320
image_height = 160
image_channel = 3
image_stack = 2
image_batch = 1
image_path = "./data"
image_format = "png"
label_path = "./data"
label_name = "pose_label.txt"


conv_dropout = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
conv_batch = [True, True, True, True, True, True, True, True, True]
conv_kernel = [(7, 7), (5, 5), (5, 5), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
conv_padding = [3, 2, 2, 1, 1, 1, 1, 1, 1]
conv_stride = [2, 2, 2, 1, 2, 1, 2, 1, 2]
conv_filter = [64, 128, 256, 256, 512, 512, 512, 512, 1024]



lstm_out = 1000
lstm_dropout = 0.5
lstm_stateful = "True"
dense_in = 1000
dense_out = 3

weight_path = "/home/dyros-vehicle/laneSeg/VO_weight1576719881.65711.h5"