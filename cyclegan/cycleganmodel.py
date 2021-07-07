from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import RandomNormal
from keras.models import Input
from keras.layers import Conv2D
from numpy.lib import stride_tricks
from keras.models import Model
from keras.optimizers import Adam 
from keras.layers import Activation
from keras.layers import Concatenate

layer = InstanceNormalization(axis=-1)

def discriminator (image_shape):
    weight_ini=RandomNormal(stddev=0.02)
    in_image= Input(shape = image_shape)
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_ini)(in_image)
    d=LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_ini)(d)
    d=InstanceNormalization(axis=-1)(d)
    d=LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_ini)(d)
    d=InstanceNormalization(axis=-1)(d)
    d=LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_ini)(d)
    d=InstanceNormalization(axis=-1)(d)
    d=LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_ini)(d)
    d=InstanceNormalization(axis=-1)(d)
    d=LeakyReLU(alpha=0.2)(d)
    patch_output= Conv2D(1, (4,4), padding='same', kernel_initializer=weight_ini)(d)
    model=Model(in_image, patch_output)
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model

def resnet(nfilters, input_layer):
    initial=RandomNormal(stddev=0.02)
    g=Conv2D(nfilters, (3,3), padding='same', kernel_initializer=initial)(input_layer)
    g=InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    g=Conv2D(nfilters, (3,3), padding='same', kernel_initializer=initial)(g)
    g=InstanceNormalization(axis=-1)(g)
    g=Concatenate()([g,input_layer])
    return g
    
def generator(image_shape, n_resnet):
    weight_ini=RandomNormal(stddev=0.02)
    in_image= Input(shape = image_shape)
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=weight_ini)(in_image)
    g=InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=weight_ini)(g)
    g=InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    g= Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=weight_ini)(g)
    g=InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    for _ in range (n_resnet):
        g=resnet(256,g)
    g=Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=weight_ini)(g)
    g=InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    g=Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=weight_ini)(g)
    g=InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    g=Conv2D(3, (7,7), padding='same', kernel_initializer=weight_ini)(g)
    g=InstanceNormalization(axis=-1)(g)
    out_image=Activation('tanh')(g)
    model=Model(in_image, out_image)
    return model 