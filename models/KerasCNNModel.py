__author__ = 'JunSong<songjun54cm@gmail.com>'
import logging
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.layers import Softmax
from keras.models import  Model
import keras.backend as K
import keras


class KerasCNNModel(object):
    def __init__(self):
        self.keras_model = None
        self.input_label = None
        self.input_feature = None
        self.probabilities = None
        self.predict_class = None
        self.predict_accuracy = None
        self.model_loss = None
        self.regularize_loss = None
        self.loss = None

    def create(self, model_config):
        logging.info('creating %s' % self.__class__.__name__)

        LearningRate = model_config['learning_rate']
        DropoutRate = model_config['dropout_rate']
        GradClip = model_config['grad_clip']

        label_input = Input(shape=(), name='input_label')
        self.input_label = label_input
        feature_input = Input(shape=(28, 28, 1), name='input_feature')
        self.input_feature = feature_input

        conv1 = Conv2D(filters=32,
                       kernel_size=(5,5),
                       strides=(1, 1),
                       padding='same',
                       activation='relu')(feature_input)

        pool1 = MaxPool2D(pool_size=(2,2),
                          strides=(2,2))(conv1)

        conv2 = Conv2D(filters=64,
                       kernel_size=(5,5),
                       strides=(1,1),
                       padding='same',
                       activation='relu')(pool1)

        pool2 = MaxPool2D(pool_size=(2,2),
                          strides=(2,2))(conv2)

        pool2_flat = Flatten()(pool2)

        dense1 = Dense(units=1024, activation='relu')(pool2_flat)
        dropout1 = Dropout(rate=DropoutRate)(dense1)
        logits = Dense(units=10)

        probabilities = Softmax(name="probabilities")(logits)
        self.probabilities = probabilities
        pred_class = K.argmax(probabilities)
        self.predict_class = pred_class
        pred_accuracy = keras.metrics.sparse_categorical_accuracy(y_true=label_input, y_pred=pred_class)
        self.predict_accuracy = pred_accuracy

        model_loss = keras.losses.sparse_categorical_crossentropy(y_true=label_input, y_pred=pred_class)
        self.model_loss = model_loss
        regularize_loss = keras.losses.get_regularization_loss()
        self.regularize_loss = regularize_loss
        total_loss = model_loss + regularize_loss
        self.loss = total_loss


    def train(self, batch_feature, batch_label):
        pass

    def predict_probability(self, feature):
        pass

    def predict_label(self, feature):
        pass

    def save(self, target_path):
        pass

    def load(self, soruce_path):
        pass

