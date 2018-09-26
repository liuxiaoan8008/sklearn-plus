#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

from sklearn.base import BaseEstimator
from sklearn_plus.utils.data_helpers import batch_iter

from .models.lstm import LSTM
from .models.bilstm import BiLSTM
from .models.bilstm_crf import BiLSTMCrf
from .models.cnn_bilstm_crf import CNNBiLSTMCrf
from .models.blstm_att_crf import BiLSTMATTCrf

from ...base import ModelMixin

import tensorflow as tf

class EntityRecongnizer(BaseEstimator, ModelMixin):
    def __init__(self, vocab_size, label_vocab_size, embedding_size=128, learning_rate=0.01, dropout_prob=0.5, batch_size=128,
                 num_epochs=10, checkpoint_dir=None, summary_dir=None, model_name='lstm'):
        super(EntityRecongnizer, self).__init__()  # call super init method first
        self.vocab_size = vocab_size
        self.label_vocab_size = label_vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_prob = dropout_prob
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir
        self.model_name = model_name
        pass

    def fit(self, X, y=None):
        # init model
        if self.model_name == 'lstm':
            self.model = LSTM(X.shape[1],self.vocab_size,self.label_vocab_size,self.embedding_size)
        elif self.model_name == 'bilstm':
            self.model = BiLSTM(X.shape[1],self.vocab_size,self.label_vocab_size,self.embedding_size)
        elif self.model_name == 'bilstm-crf':
            self.model = BiLSTMCrf(X.shape[1],self.vocab_size,self.label_vocab_size,self.embedding_size)
        elif self.model_name == 'cnn':
            self.model = CNNBiLSTMCrf(X.shape[1],self.vocab_size,self.label_vocab_size,self.embedding_size)
        elif self.model_name == 'att':
            self.model = BiLSTMATTCrf(X.shape[1],self.vocab_size,self.label_vocab_size,self.embedding_size)


            # define train op
        # need add global_step
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.model.loss,
                                                                                    global_step=self.global_step)

        # define summary dictionary if needed.
        summaries_dict = {
            'loss': self.model.loss
        }

        # init variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # create batch generator
        batches = batch_iter(list(zip(X, y)), self.batch_size, self.num_epochs)

        # validation
        def validation(x_batch, y_batch):
            feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch,
                self.model.dropout_keep_prob: 1
            }

            predictions = []
            if self.model_name == 'bilstm-crf' or 'cnn':

                logits, transition_params,sequence_lengths = self.sess.run(
                    [self.model.logits,self.model.transition_param, self.model.sequence_lengths], feed_dict)
                for logit,sequence_length in zip(logits, sequence_lengths):
                    logit = logit[:sequence_length]
                    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
                    predictions.append(viterbi_sequence)
            else:
                predictions = self.sess.run(self.model.predictions, feed_dict)

            right_count = 0
            all_count = 0
            for predict_label, true_label in zip(predictions, y_batch):
                for p_l,t_l in zip(predict_label,true_label):
                    if t_l == 0:
                        break
                    if p_l == t_l:
                        right_count += 1
                    all_count += 1
            return right_count*1.0/all_count

        # define train step
        def train_step(x_batch, y_batch, summaries_dict=None):
            feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch,
                self.model.dropout_keep_prob: self.dropout_prob
            }

            _, step, loss = self.sess.run(
                [train_op, self.global_step, self.model.loss],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

            if step % 50 == 0:
                acc = validation(x_batch, y_batch)
                print("{}: step {}, loss {:g}, valid acc: {:g}".format(time_str, step, loss, acc))

                # if summary is needed
                if self.summary_dir:
                    self.summaries(self.summary_dir, feed_dict, summaries_dict)

                # if checkpoint is needed
                if self.checkpoint_dir:
                    self.save(self.checkpoint_dir)

        # train
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
        return logits.tolist()

