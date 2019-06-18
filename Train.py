 
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

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE =50
LR = 1e-3
MODEL_NAME = 'Model'

def create_label(image_name):
    word_label = image_name.split('.')[-3]
    if word_label == 'Books':
        return  np.array([1, 0, 0, 0, 0, 0])
    elif word_label == 'Glasses':
        return  np.array([0, 1, 0, 0, 0, 0])
    elif word_label == 'Laptop':
        return  np.array([0, 0, 1, 0, 0, 0])
    elif word_label == 'Mouse':
        return  np.array([0, 0, 0, 1, 0, 0])
    elif word_label == 'Pen':
        return  np.array([0, 0, 0, 0, 1, 0])
    else:
        return  np.array([0, 0, 0, 0, 0, 1])

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_data is not None:
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img_data), create_label(img)])
        else: print("File Not Found")

    shuffle(training_data)
    np.save('data-train.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_data is not None:
            img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img_data), img_num])
        else: print("File Not Found")

    shuffle(testing_data)
    np.save('data-test.npy', testing_data)
    return testing_data

#train_data = create_train_data()

test_data = create_test_data()
#test_data = create_test_data()
#test_data = create_test_data()
