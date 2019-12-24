from keras import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Concatenate, Flatten, LeakyReLU, LSTM, ZeroPadding2D, Lambda
from keras.backend import reshape, is_keras_tensor, concatenate
from keras.optimizers import Adam
from keras.activations import relu, sigmoid, tanh
from keras.initializers import he_normal
from keras.regularizers import l1_l2
import config_dat as cf
import numpy as np
from Config import Vehicle

def ConvBlock(tensor, batchnorm, filter, kernel_size, strides, padding, dropout):
    if batchnorm:
        padded_tensor = ZeroPadding2D(padding)(tensor)
        conv = Conv2D(filter, kernel_size, strides=strides, padding='valid', bias_initializer='zero', kernel_initializer='he_normal', kernel_regularizer=l1_l2(0.01, 0.01))(padded_tensor)
        batch = BatchNormalization(axis=-1)(conv)
        relu = LeakyReLU(0.1)(batch)
        drop = Dropout(dropout)(relu)
        return drop
    else:
        padded_tensor = ZeroPadding2D(padding)(tensor)
        conv = Conv2D(filter, kernel_size, strides=strides, padding='valid', bias_initializer='zero', kernel_initializer='he_normal')(padded_tensor)
        relu = LeakyReLU(0.1)(conv)
        drop = Dropout(dropout)(relu)
        return drop



class DRVO():
    def __init__(self):
        # Input shape : (batch, image stack, height, width, channel)
        self.input = Input(batch_shape=(cf.image_batch, cf.image_stack, cf.image_height, cf.image_width, cf.image_channel))
        self.stacked_input = Lambda(lambda tensor: concatenate([tensor[:, 1:, :, :, :], tensor[:, :-1, :, :, :]], axis=-1))(self.input)
        # self.stacked_input = Concatenate(axis=-1)([self.input[:, 1:, :, :, :], self.input[:, :-1, :, :, :]])
        self.stack_len = self.stacked_input.shape[1]
        self.reshape = Lambda(lambda tensor: reshape(tensor, shape=(cf.image_batch * self.stack_len, cf.image_height, cf.image_width, cf.image_channel * 2)))(self.stacked_input)
        # print(self.reshape.shape)
        self.conv1 = ConvBlock(self.reshape, cf.conv_batch[0], cf.conv_filter[0], cf.conv_kernel[0], cf.conv_stride[0], cf.conv_padding[0], cf.conv_dropout[0])
        self.conv2 = ConvBlock(self.conv1, cf.conv_batch[1], cf.conv_filter[1], cf.conv_kernel[1], cf.conv_stride[1], cf.conv_padding[1], cf.conv_dropout[1])
        self.conv3 = ConvBlock(self.conv2, cf.conv_batch[2], cf.conv_filter[2], cf.conv_kernel[2], cf.conv_stride[2], cf.conv_padding[2], cf.conv_dropout[2])
        self.conv3_1 = ConvBlock(self.conv3, cf.conv_batch[3], cf.conv_filter[3], cf.conv_kernel[3], cf.conv_stride[3], cf.conv_padding[3], cf.conv_dropout[3])
        self.conv4 = ConvBlock(self.conv3_1, cf.conv_batch[4], cf.conv_filter[4], cf.conv_kernel[4], cf.conv_stride[4], cf.conv_padding[4], cf.conv_dropout[4])
        self.conv4_1 = ConvBlock(self.conv4, cf.conv_batch[5], cf.conv_filter[5], cf.conv_kernel[5], cf.conv_stride[5], cf.conv_padding[5], cf.conv_dropout[5])
        self.conv5 = ConvBlock(self.conv4_1, cf.conv_batch[6], cf.conv_filter[6], cf.conv_kernel[6], cf.conv_stride[6], cf.conv_padding[6], cf.conv_dropout[6])
        self.conv5_1 = ConvBlock(self.conv5, cf.conv_batch[7], cf.conv_filter[7], cf.conv_kernel[7], cf.conv_stride[7], cf.conv_padding[7], cf.conv_dropout[7])
        self.conv6 = ConvBlock(self.conv5_1, cf.conv_batch[8], cf.conv_filter[8], cf.conv_kernel[8], cf.conv_stride[8], cf.conv_padding[8], cf.conv_dropout[8])
        self.conv_reshape = Lambda(lambda tensor: reshape(tensor, shape=(cf.image_batch, self.stack_len, -1)))(self.conv6)


        #LSTM (Temporal Context Extraction)
        self.lstm_1 = LSTM(cf.lstm_out, use_bias=True, unit_forget_bias=True, dropout=cf.lstm_dropout, stateful=cf.lstm_stateful, bias_initializer='zero', kernel_initializer= 'orthogonal', return_sequences=True)(self.conv_reshape)
        self.lstm_2 = LSTM(cf.lstm_out, use_bias=True, unit_forget_bias=True, dropout=cf.lstm_dropout, stateful=cf.lstm_stateful, bias_initializer='zero', kernel_initializer= 'orthogonal', return_sequences=True)(self.lstm_1)

        # Current Velocity & Heading Prediction
        self.output = Dense(cf.dense_out, input_shape=(cf.image_batch, cf.dense_in, ), bias_initializer='zero')(self.lstm_2)
        self.model = Model(inputs=self.input, outputs=self.output)






