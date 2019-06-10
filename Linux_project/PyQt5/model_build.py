import os
import tensorflow as tf
from keras import optimizers
from keras.models import Model
from keras.layers import *
from keras.engine.topology import Layer
from keras.applications.vgg16 import VGG16
from squeezenet import SqueezeNet
from keras import activations
from keras import backend as K

from keras.utils import plot_model, to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np
from keras.models import load_model
from keras import metrics

import PIL
from itertools import chain
import psutil  # check memory usage

# label definition
TRASH_DICT = {
    '0': 'glass',
    '1': 'paper',
    '2': 'cardboard',
    '3': 'plastic',
    '4': 'metal',
    '5': 'trash',
    '6': 'battery',
    '7': 'organic',  # 湿垃圾（有机垃圾）
    '8': 'harmful'  # 有害垃圾
}

TRASH_KIND = {
    'glass': 'Recyclable Waste',
    'paper': 'Recyclable Waste',
    'cardboard': 'Recyclable Waste',
    'plastic': 'Recyclable Waste',
    'metal': 'Recyclable Waste',
    'trash': 'Residual Waste',
    'organic': 'Household Waste',
    'battery': 'Hazardous Waste',
    'harmful': 'Hazardous Waste'
}
TRASH_DICT_R = {v: k for k, v in TRASH_DICT.items()}

lambda_val = 0.5
# 上margin与下margin的参数值
m_plus = 0.9
m_minus = 0.1
# 准备训练数据
batch_size = 128
channel = 3
capsule_output_dims = 32
epsilon = 1e-9
batch_size = 128


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule=10, dim_capsule=32, routings=3, share_weights=True, activation='squash', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
            # print("self.W shape",self.W.shape)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            # 矩阵点乘加扩展？
            u_hat_vecs = K.conv1d(u_vecs, self.W)
            # print("After conv1d u_hat_vecs shape",u_hat_vecs.shape)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        # print("After reshape u_hat_vecs shape",u_hat_vecs.shape)
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # print("After permute u_hat_vecs shape",u_hat_vecs.shape)
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        # print("b shape",b.shape)
        for i in range(self.routings):
            c = softmax(b, 1)
            # print("c shape",c.shape)
            o = K.batch_dot(c, u_hat_vecs, [2, 2])
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                # o = K.l2_normalize(o, -1)
                o = squash(o)
                # print("After l2 ",o.shape)
                b = K.batch_dot(o, u_hat_vecs, [2, 3])
                # print("batch_dot",b.shape)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def squash(self, vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -1, keep_dims=True)
        scalar_factor = vec_squared_norm / (0.5 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return (vec_squashed)


def my_loss(y_true, y_pred):
    return y_true * K.relu(m_plus - y_pred) ** 2 + lambda_val * (1 - y_true) * K.relu(y_pred - m_minus) ** 2


def non_local_sa_layer(input):
    ip_shape = K.int_shape(input)
    batchsize, dim1, dim2, channels = ip_shape
    intermediate_dim = channels // 2
    # bs*H*W*C/2
    # θpath
    theta = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                   name='SA_theta_conv')(input)
    theta = Reshape((-1, intermediate_dim), name='SA_theta_reshape')(theta)

    # φpath
    phi = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                 name='SA_phi_conv')(input)
    phi = Reshape((-1, intermediate_dim), name='SA_phi_reshape')(phi)

    f = dot([theta, phi], axes=2, name='SA_theta_dot_phi')
    size = K.int_shape(f)
    # f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)
    f = Activation('softmax', name='SA_softmax')(f)

    # g path
    g = Conv2D(intermediate_dim, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               name='SA_g_conv')(input)
    g = Reshape((-1, intermediate_dim), name='SA_g_reshape')(g)

    y = dot([f, g], axes=[2, 1], name='SA_final_dot')

    # return to the same shape as input
    y = Conv1D(channels, (1), padding='same', use_bias=False, kernel_initializer='he_normal', name='SA_output_conv')(y)
    # y=Reshape((dim1,dim2,channels),name='SA_output_reshape')(y)
    return y


def build_prune_model():
    try:
        del model
        print('clear former model')
    except:
        print('No former model')

    input_image = Input(shape=(224, 224, 3))
    vgg16 = SqueezeNet(include_top=False, input_tensor=input_image)
    for i in range(2):
        vgg16.layers.pop()

    for layer in vgg16.layers[2:58]:
        layer.trainable = False

    # pre_trained vgg16 model to extract the feature maps
    caps1_num_classes = 64
    caps1_capsule_output = 128
    caps2_num_classes = len(TRASH_DICT)
    caps2_capsule_output = 16

    cnn = non_local_sa_layer(vgg16.layers[-1].output)
    cnn = Dropout(0.2)(cnn)
    capsule1 = Capsule(caps1_num_classes, caps1_capsule_output, share_weights=True, name='Capsule_net_1')(cnn)
    capsule1 = Dropout(0.1)(capsule1)
    capsule2 = Capsule(caps2_num_classes, caps2_capsule_output, share_weights=True, name='Capsule_net_2')(capsule1)
    # capsule2=Dropout(0.1)(capsule2)
    caps_output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(caps2_num_classes,), name='softmax')(
        capsule2)
    # using sgd optimizer
    model = Model(inputs=input_image, outputs=caps_output)
    model.compile(loss=my_loss,
                  optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    try:
        model.load_weights(
            './my_prune_model_weights_addaug_lr_0.1.h5')
        print('load weights successed!\n')
    except:
        print("Load pretrained weights failed!")

    return model


def build_general_model():
    try:
        del model
        print('clear former model')
    except:
        print('No former model')
    # design the model architecture
    input_image = Input(shape=(224, 224, 3))
    vgg16 = VGG16(
        weights='./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False, pooling='avg', input_tensor=input_image)
    for i in range(2):
        vgg16.layers.pop()
    for layer in vgg16.layers[2:15]:
        # print layer.name,layer
        layer.trainable = False

    # pre_trained vgg16 model to extract the feature maps
    caps1_num_classes = 16
    caps1_capsule_output = 64
    caps2_num_classes = len(TRASH_DICT)
    caps2_capsule_output = 16

    pool = MaxPool2D((2, 2), strides=2, padding='valid', name='block5_Maxpooling2D')(vgg16.layers[-1].output)
    # cnn=Conv2D(256,(1,1),padding='same', use_bias=False, kernel_initializer='he_normal',name='block6_conv1')(pool)
    cnn = Dropout(0.3)(pool)
    cnn = non_local_sa_layer(cnn)

    capsule1 = Capsule(caps1_num_classes, caps1_capsule_output, share_weights=True, name='Capsule_net_1')(cnn)
    capsule1 = Dropout(0.1)(capsule1)
    capsule2 = Capsule(caps2_num_classes, caps2_capsule_output, share_weights=True, name='Capsule_net_2')(capsule1)
    # capsule2=Dropout(0.1)(capsule2)
    caps_output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(caps2_num_classes,), name='softmax')(
        capsule2)
    # using sgd optimizer
    model = Model(inputs=input_image, outputs=caps_output)
    model.compile(loss=my_loss,
                  optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    try:
        model.load_weights(
            './my_model_weights_addaug_lr_0.1.h5')
        print('load weights successed!\n')
    except:
        print("Load pretrained weights failed!")

    return model
