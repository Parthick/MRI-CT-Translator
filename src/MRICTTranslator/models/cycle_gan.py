from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    Conv2DTranspose,
    LeakyReLU,
    Activation,
    Concatenate,
)
from matplotlib import pyplot
from MRICTTranslator.utils.common import InstanceNormalization


# discriminator model (70x70 patchGAN)
# C64-C128-C256-C512
# After the last layer, conv to 1-dimensional output, followed by a Sigmoid function.
# The “axis” argument is set to -1 for instance norm. to ensure that features are normalized per feature map.
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64: 4x4 kernel Stride 2x2
    d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
        in_image
    )
    d = LeakyReLU(alpha=0.2)(d)
    # C128: 4x4 kernel Stride 2x2
    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256: 4x4 kernel Stride 2x2
    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512: 4x4 kernel Stride 2x2
    # Not in the original paper. Comment this block if you want.
    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer : 4x4 kernel but Stride 1x1
    d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
    # define model
    model = Model(in_image, patch_out)
    # compile model
    # The model is trained with a batch size of one image and Adam opt.
    # with a small learning rate and 0.5 beta.
    # The loss for the discriminator is weighted by 50% for each model update.
    # This slows down changes to the discriminator relative to the generator model during training.
    model.compile(
        loss="mse", optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5]
    )
    return model


# generator a resnet block to be used in the generator
# residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layers.
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first convolutional layer
    g = Conv2D(n_filters, (3, 3), padding="same", kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


# define the  generator model - encoder-decoder type architecture

# c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
# dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2.
# Rk denotes a residual block that contains two 3 × 3 convolutional layers
# uk denotes a 3 × 3 fractional-strided-Convolution InstanceNorm-ReLU layer with k filters and stride 1/2

# The network with 6 residual blocks consists of:
# c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

# The network with 9 residual blocks consists of:
# c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,R256,R256,R256,u128, u64,c7s1-3


def define_generator(image_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7, 7), padding="same", kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    # d128
    g = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    # d256
    g = Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv2DTranspose(
        128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init
    )(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    # u64
    g = Conv2DTranspose(
        64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init
    )(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    # c7s1-3
    g = Conv2D(3, (7, 7), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation("tanh")(g)
    # define model
    model = Model(in_image, out_image)
    return model


# define a composite model for updating generators by adversarial and cycle loss
# We define a composite model that will be used to train each generator separately.
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # Make the generator of interest trainable as we will be updating these weights.
    # by keeping other models constant.
    # Remember that we use this same function to train both generators,
    # one generator at a time.
    g_model_1.trainable = True
    # mark discriminator and second generator as non-trainable
    d_model.trainable = False
    g_model_2.trainable = False

    # adversarial loss
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity loss
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # cycle loss - forward
    output_f = g_model_2(gen1_out)
    # cycle loss - backward
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    # define model graph
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    # define the optimizer
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(
        loss=["mse", "mae", "mae", "mae"], loss_weights=[1, 5, 10, 10], optimizer=opt
    )
    return model
