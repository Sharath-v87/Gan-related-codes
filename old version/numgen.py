from keras import Sequential
from keras.models import load_model
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Flatten, Dense, Reshape
from numpy import array as ar
from numpy import asarray as asar
from numpy import random
from numpy import append
from numpy import ones
from numpy import amax

# def discriminator():
#     model = Sequential()
#     model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28,28,3)))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))
#     return model

def generator():
    model = Sequential()
    n_nodes = 64 * 8 * 8
    model.add(Dense(n_nodes, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Reshape((8, 8, 64)))
    model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Conv2D(1, (3,3), activation='tanh', padding='same'))
    return model


def gan(disc, gen):
    disc.trainable = False
    model = Sequential()
    model.add(gen)
    model.add(disc)
    return model


disc = load_model('digitsocr.h5')
disc.trainable = False
#gen = generator()
#g = gan(disc, gen)
g = load_model('g.h5')
gen = load_model('gen.h5')
n_batch = 16
latent_dim = 100

def genlat(dim, n):
    l = random.randint(0, high=9, size=(n, dim))
    return l
y = asar([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
y_gan = []
for i in range(16):
    y_gan.append(y)
y_gan=asar(y_gan)
print(y_gan.shape)
g.compile(optimizer='Adamax', loss = 'KLDivergence')
prob = 0
s = 10
i = 0
while prob<0.95:
    X_gan = genlat(latent_dim, n_batch)
    g_loss = g.train_on_batch(X_gan, y_gan)
    prob = g.predict(X_gan)
    prob = prob[0][1]
    i += 1
    if i%100 == 0:
        print(prob)
while s>0.2:
    X_gan = genlat(latent_dim, n_batch)
    g_loss = g.train_on_batch(X_gan, y_gan)
    prob = g.predict(X_gan)
    prob1 = prob[0][1]
    prob0 = prob[0][0]
    prob2 = prob[0][2]
    prob3 = prob[0][3]
    prob4 = prob[0][4]
    prob5 = prob[0][5]
    prob6 = prob[0][6]
    prob7 = prob[0][7]
    prob8 = prob[0][8]
    prob9 = prob[0][9]
    s = prob0 + prob2 + prob3 + prob4 + prob5 + prob6 + prob7 + prob8 + prob9
    i += 1
    if i%100 == 0:
        print(prob)
        print(s)
gen.save('gen.h5')
g.save('g.h5')