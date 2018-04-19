__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import numpy as np
import tensorflow as tf
import random
import os
import logging


class TFCNNModel(object):
    def __init__(self):
        self.input_feature = None
        self.input_label = None
        self.probabilities = None
        self.predict_class = None
        self.predict_accuracy = None
        self.model_loss = None
        self.regularize_loss = None
        self.loss = None
        self.train_op = None
        self.session = None

    def create(self, model_config):
        logging.info('creating %s' % self.__class__.__name__)

        LearningRate = model_config['learning_rate']
        DropoutRate = model_config['dropout_rate']
        GradClip = model_config['grad_clip']
        # Input Layer
        label_input_layer = tf.placeholder(tf.int64, [None], name='input_label')
        self.input_label = label_input_layer
        fea_input_layer = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_feature')
        self.input_feature = fea_input_layer

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=fea_input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=DropoutRate)

        logits = tf.layers.dense(inputs=dropout, units=10)

        probabilities = tf.nn.softmax(logits, name='probabilities')
        self.probabilities = probabilities
        pred_class = tf.argmax(input=logits, axis=1, name='predict_class')
        self.predict_class = pred_class
        pred_accuracy = tf.reduce_mean(tf.to_float(tf.equal(pred_class, label_input_layer)), name='predict_accuracy')
        self.predict_accuracy = pred_accuracy

        # Calculate Loss
        model_loss = tf.losses.sparse_softmax_cross_entropy(labels=label_input_layer, logits=logits)
        self.model_loss = model_loss
        regularize_loss = tf.losses.get_regularization_loss()
        self.regularize_loss = regularize_loss
        total_loss = model_loss + regularize_loss
        self.loss = total_loss

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LearningRate)
        # train_op = optimizer.minimize(loss=total_loss)
        grad_vals = optimizer.compute_gradients(loss=total_loss)
        capped_grad_vals = [(tf.clip_by_value(grad, -GradClip, GradClip), var) for grad, var in grad_vals]
        train_op = optimizer.apply_gradients(capped_grad_vals)
        self.train_op = train_op

        if self.session is None:
            logging.info('initialize variables.')
            self.session = tf.Session()
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.session.run(init)

    def train(self, batch_feature, batch_label):
        _, model_loss, regu_loss, total_loss = self.session.run(
            (self.train_op, self.model_loss, self.regularize_loss, self.loss),
            feed_dict={
                self.input_feature:batch_feature,
                self.input_label:batch_label
            })
        return total_loss, model_loss, regu_loss

    def predict_probability(self, feature):
        probs = self.session.run(
            self.probabilities,
            feed_dict={
                self.input_feature:feature
            }
        )
        return probs

    def predict_label(self, feature):
        label = self.session.run(
            self.predict_class,
            feed_dict={
                self.input_feature: feature
            }
        )
        return label

    def save(self, target_path):
        builder = tf.saved_model.builder.SavedModelBuilder(target_path)
        builder.add_meta_graph_and_variables(self.session, [tf.saved_model.tag_constants.TRAINING], clear_device=True)
        model_path = builder.save()
        logging.info('save model into %s' % model_path)

    def save_model_saver(self, target_path):
        saver = tf.train.Saver()
        saver.save(self.session, os.path.join(target_path, 'model'))
        logging.info('save model into %s' % target_path)


    def load(self, source_path):
        if self.session is None:
            self.session = tf.Session()
        tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.TRAINING], source_path)
        graph = tf.get_default_graph()

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        self.input_label = graph.get_tensor_by_name("input_label:0")
        self.input_feature = graph.get_tensor_by_name("input_feature:0")
        self.probabilities = graph.get_tensor_by_name("probabilities:0")
        self.predict_class = graph.get_tensor_by_name("predict_class:0")
        self.predict_accuracy = graph.get_tensor_by_name("predict_accuracy:0")


    def load_model_saver(self, source_path):
        if self.session is None:
            self.session = tf.Session()
        meta_path = os.path.join(source_path, 'model.meta')
        saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
        saver.restore(self.session, tf.train.latest_checkpoint(source_path))

        graph = tf.get_default_graph()
        self.input_label = graph.get_tensor_by_name("input_label:0")
        self.input_feature = graph.get_tensor_by_name("input_feature:0")
        self.probabilities = graph.get_tensor_by_name("probabilities:0")
        self.predict_class = graph.get_tensor_by_name("predict_class:0")
        self.predict_accuracy = graph.get_tensor_by_name("predict_accuracy:0")