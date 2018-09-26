#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn_plus.utils.data_helpers import batch_iter

from .models.cnn_lstm import CNN_LSTM  # OPTION 0
from .models.lstm_cnn import LSTM_CNN  # OPTION 1
from .models.cnn import CNN  # OPTION 2 (Model by: Danny Britz)
from .models.lstm import LSTM  # OPTION 3
from ...base import ModelMixin  #

import tensorflow as tf
import numpy as np


# Source: https://github.com/pmsosa/CS291K
class TextClassifier(BaseEstimator, ClassifierMixin, ModelMixin):

    def __init__(self, vocab_size, checkpoint_dir=None, summary_dir=None, MODEL_TO_RUN=0, embedding_dim=32,
                 filter_sizes=[3, 4, 5], num_filters=32, dropout_prob=0.5, l2_reg_lambda=0.0, batch_size=128,
                 num_epochs=10):
        super(TextClassifier, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir
        self.vocab_size = vocab_size

        self.MODEL_TO_RUN = MODEL_TO_RUN
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_prob = dropout_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def fit(self, X, y=None):
        if (self.MODEL_TO_RUN == 0):
            self.model = CNN_LSTM(X.shape[1], y.shape[1], self.vocab_size,
                                  self.embedding_dim, self.filter_sizes, self.num_filters, self.l2_reg_lambda)
        elif (self.MODEL_TO_RUN == 1):
            self.model = LSTM_CNN(X.shape[1], y.shape[1], self.vocab_size,
                                  self.embedding_dim, self.filter_sizes, self.num_filters, self.l2_reg_lambda)
        elif (self.MODEL_TO_RUN == 2):
            self.model = CNN(X.shape[1], y.shape[1], self.vocab_size,
                             self.embedding_dim, self.filter_sizes, self.num_filters, self.l2_reg_lambda)
        elif (self.MODEL_TO_RUN == 3):
            self.model = LSTM(X.shape[1], y.shape[1], self.vocab_size,
                              self.embedding_dim)
        else:
            print("PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n")
            exit();

        # Define Training procedure
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # init variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        summaries_dict = {
            "loss": self.model.loss,
            "accuracy": self.model.accuracy
        }

        # TRAINING STEP
        def train_step(x_batch, y_batch, summaries_dict=None):
            feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch,
                self.model.dropout_keep_prob: self.dropout_prob
            }

            _, step, loss, accuracy = self.sess.run(
                [train_op, self.global_step, self.model.loss, self.model.accuracy],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            if step % 50 == 0:
                # if summary is needed
                if self.summary_dir:
                    self.summaries(self.summary_dir, feed_dict, summaries_dict)

                # if checkpoint is needed
                if self.checkpoint_dir:
                    self.save(self.checkpoint_dir)

        # CREATE THE BATCHES GENERATOR
        batches = batch_iter(list(zip(X,y)), self.batch_size, self.num_epochs)

        # TRAIN FOR EACH BATCH
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch, summaries_dict=summaries_dict)

    def predict(self, X):
        feed_dict = {
            self.model.input_x: X,
            self.model.dropout_keep_prob: 1
        }
        prediction_result = self.sess.run(
            self.model.predictions,
            feed_dict)
        return prediction_result

    def predict_proba(self, X):
        feed_dict = {
            self.model.input_x: X,
            self.model.dropout_keep_prob: 1
        }
        logits = self.sess.run(
            self.model.logits,
            feed_dict)
        return tf.nn.softmax(logits).tolist()
