import os

import tensorflow as tf
from tensorflow import keras
import imageio
import skimage
import skimage.transform
import numpy as np
import keras
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split

#change the model name to run for second dataset
new_model = tf.keras.models.load_model('D:\\second term\\dl\\Assignment2\\resnet_model_scene-15.h5')

# Check its architecture
new_model.summary()

def imread(path):
    img = imageio.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.transpose(np.array([img, img, img]), (2, 0, 1))
    return img
    
path = 'D:\\second term\\dl\\Assignment2\\15-Scene'
valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))

categories = sorted(os.listdir(path))
ncategories = len(categories)
imgs = []
labels = []
# LOAD ALL IMAGES 
for i, category in enumerate(categories):
    iter = 0
    for f in os.listdir(path + "/" + category):
        if iter == 0:
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_exts:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            img = skimage.transform.resize(imread(fullpath), [32,32, 3])
            # img = img.astype('float32')
            # img[:,:,0] -= 123.68
            # img[:,:,1] -= 116.78
            # img[:,:,2] -= 103.94
            imgs.append(img) # NORMALIZE IMAGE 
            label_curr = i
            labels.append(label_curr)
        #iter = (iter+1)%10;
print ("Num imgs: %d" % (len(imgs)))
print ("Num labels: %d" % (len(labels)) )
print (ncategories)


seed = 7
np.random.seed(seed)

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.3)
x_train = np.stack(x_train, axis=0)
y_train = np.stack(y_train, axis=0)
x_test = np.stack(x_test, axis=0)
y_test = np.stack(y_test, axis=0)
print ("Num train_imgs: %d" % (len(x_train)))
print ("Num test_imgs: %d" % (len(x_test)))
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

scores = new_model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0] / 1000)
print('Test accuracy:', scores[1] * 10)