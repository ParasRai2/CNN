import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from keras.models import load_model
import threading
import time

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'Model'

#Model Start
def create_model():
        
        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 256, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 6, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name ='targets')

        return tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)


model = create_model()  
if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
#Model Ended


def create_test_data(img):
        testing_data = []
        img_num = 0
        if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                testing_data.append([np.array(img), img_num])
                img_num = img_num + 1

        shuffle(testing_data)
        return testing_data





fig = plt.figure(figsize=(16, 12))

tf.reset_default_graph()

for num, data in enumerate(test_data[:16]):

        img_num = data[1]
        img_data = data[0]

        image = cv2.imread("cat.jpg") 
        image = cv2.resize(image,(400,400))
        tmp = image
        stepSize =10
        (w_width, w_height) = (100, 100) 
        for x in range(0, image.shape[1] - w_width , stepSize):
                for y in range(0, image.shape[0] - w_height, stepSize):
                        
                        window = image[x: x + w_width, y: y + w_height, :]
                        cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2) 


        im = Image.open('0.png').convert('L')
        im = im.crop((1, 1, 98, 33))
        im.save('_0.png')


        y = fig.add_subplot(4, 4, num+1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        cat = model_out[0]
        dog = model_out[1]
        
        if np.argmax(model_out) == 0:
                str_label = 'Books'
        elif np.argmax(model_out) == 1:
                str_label = 'Glasses'
        elif np.argmax(model_out) == 2:
                str_label = 'Laptop'
        elif np.argmax(model_out) == 3:
                str_label = 'Mouse'
        elif np.argmax(model_out) == 4:
                str_label = 'Pen'
        elif np.argmax(model_out) == 5:
                str_label = 'Watch'
        
        y.imshow(orig, cmap = 'gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)


plt.show()











    

    
    
