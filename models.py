'''
BCINet
© Avinash K Singh 
https://github.com/thinknew/bcinet
Licensed under MIT License
'''

####################################################################
### Original code for EEGNet, ShallowConvnet, and DeepConvNet is taken
### from: https://github.com/vlawhern/arl-eegmodels
####################################################################

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, \
    SeparableConv2D, DepthwiseConv2D, BatchNormalization, SpatialDropout2D, Input, Flatten, GaussianNoise, ConvLSTM2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
from tensorflow.keras import Input, layers


tf.disable_v2_behavior()


# sess = tf.Session()
def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    # with tf.Graph().as_default() as g:
    input1 = Input(shape=(1, Chans, Samples))

 
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False, data_format='channels_first')(input1)

    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             data_format='channels_first',
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)  # changed by avinash
    block1 = Activation('elu')(block1)

    block1 = AveragePooling2D((1, 4), data_format='channels_first', )(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),  # changed by avinash
                             data_format='channels_first', use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8), data_format='channels_first', )(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten', data_format='channels_first')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)



def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


# def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
def ShallowConvNet(nb_classes, Chans=64, Samples=128,
                   dropoutRate=0.5, kernLength=64, F1=8,
                   D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)


    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7), data_format='channels_first')(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten(data_format='channels_first')(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)



def DeepConvNet(nb_classes, Chans=64, Samples=128,
                dropoutRate=0.5, kernLength=64, F1=8,
                D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'):
    # start the model
    input_main = Input((1, Chans, Samples))

    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples), padding='same',
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(input_main)

    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)), data_format='channels_first')(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), data_format='channels_first')(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten(data_format='channels_first')(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def BCINet(nb_classes, Chans=64, Samples=128,
                   dropoutRate=0.5, kernLength=64, F1=8,
                   D=2, F2=16, norm_rate=0.25, EnK=True, dropoutType='Dropout'): # - 17th-Sept-2019

    input_tensor = Input(shape=(1, Chans, Samples))  # 1  60 150
    xx = 8
    dropout=0.25
    x = layers.Conv2D(xx, (1, kernLength), padding='same',data_format='channels_first')(input_tensor)  # Conv2D
    x = layers.Activation('elu')(x)
    x = layers.BatchNormalization()(x)

    x1 = layers.Conv2D(xx, (1, (int)(kernLength/2)), padding='same',data_format='channels_first')(input_tensor)  # Conv2D
    x1 = layers.Activation('elu')(x1)
    x1 = layers.BatchNormalization()(x1)

    x = layers.concatenate([x1, x], axis=1)


    squeeze = layers.GlobalAveragePooling2D(data_format='channels_first')(x)
    excitation = Dense(xx)(squeeze)
    excitation = layers.Activation('elu')(excitation)
    excitation = Dense(xx*2)(excitation)
    excitation = layers.Activation('sigmoid')(excitation)
    excitation = layers.Reshape((xx*2,1,1))(excitation)

    scale = layers.multiply([x, excitation])
    x = scale

    x = layers.SeparableConv2D(xx*4, (Chans, 3),data_format='channels_first')(x)
    x = layers.Activation('elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling2D((1, 4),data_format='channels_first')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.SeparableConv2D(xx*4, (1, Chans),data_format='channels_first')(x)  # SeparableConv2D
    x = layers.Activation('elu')(x)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.SeparableConv2D(xx*8, (1, Chans//2),data_format='channels_first')(x)  # SeparableConv2D
    x = layers.Activation('elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten(data_format='channels_first')(x)
    x = layers.Dropout(dropout)(x)

    class_prediction = layers.Dense(nb_classes, activation='softmax')(x)
    return Model(inputs=input_tensor, outputs=class_prediction)
