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


tf.reset_default_graph()


def fliph(imgg):
    img2= np.zeros([480, 640, 3], np.uint8)
    for i in range(640):
        img2[:,i]=imgg[:,640-i-1]
    return img2

def showinfo():
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crop_img = gray[80: 400, 120: 520]
        img_data = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        pos = np.argmax(model_out)
        if model_out[pos]<0.8:
            str_label = "None"
        else:
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
                
        data = str_label
        print(model_out[pos])
        cv2.rectangle(img, (120, 80), (520, 400), (0, 255, 0), 2)   

        cv2.imshow("Croped", crop_img[::])
        img = fliph(img)

        #img = cv2.medianBlur(img,5)
        img = cv2.bilateralFilter(img,9,75,75)
        
        cv2.putText(img, data, (120,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("My", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:

            cap.release()
            cv2.destroyAllWindows()
            exit(0)

showinfo()
