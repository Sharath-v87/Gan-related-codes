from os import listdir
from numpy import vstack
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

def load_images (path, size=(256,256)):
    img_list=list()
    for file in listdir(path):
        pixels = load_img(path+file, target_size=size)
        pixels = img_to_array(pixels)
        img_list.append(pixels)
    return asarray(img_list)

path = 'horse2zebra/'
dataA1 = load_images(path+"trainA/")
dataA2 = load_images(path+"testA/")
dataA = vstack((dataA1, dataA2))
print('A', dataA.shape)
dataB1 = load_images(path+"trainB/") 
dataB2 = load_images(path+"testB/")
dataB = vstack((dataB1, dataB2))
print('B', dataB.shape)
filename="datanumpcomp.npz"
savez_compressed(filename,dataA,dataB)
print("compressed file",filename)