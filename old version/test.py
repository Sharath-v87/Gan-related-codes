from numpy import random
from numpy import reshape
from numpy import amax
from keras.models import load_model
from cv2 import cv2

gen = load_model('gen.h5')
g = load_model('g.h5')
l = random.randint(0, high=9, size=(1, 100))
y = gen.predict(l)
prob = g.predict(l)
y = reshape(y, (32, 32, 1))
y = y*255
y = cv2.resize(y, (320, 320))
print(prob)
cv2.imwrite('output.jpg', y)