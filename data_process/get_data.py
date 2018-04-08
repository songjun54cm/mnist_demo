__author__ = 'JunSong<songjun54cm@gmail.com>'
from tensorflow.examples.tutorials.mnist import input_data
import os

from settings import DATA_HOME

data_folder = os.path.join(DATA_HOME, 'mnist')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = input_data.read_data_sets(data_folder)

import numpy as np
from mnist import MNIST
mndata = MNIST(data_folder)
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

train_images = np.reshape(np.array(train_images), (len(train_images), 28, 28))
test_images = np.reshape(np.array(test_images), (len(test_images), 28, 28))

datas = {
    'train_images': train_images,
    'train_labels': train_labels,
    'test_images': test_images,
    'test_labels': test_labels
}
