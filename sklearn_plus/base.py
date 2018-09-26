#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes for all neural networks."""
import os
import tensorflow as tf
import json


###############################################################################
class ModelMixin(object):
    """Mixin class for all neural networks in sklearn_plus."""

    def __init__(self):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def save(self, out_dir):
        """save model weights to out_dir.

        This method save a tensorflow model's weights to specified directory.

        Parameters
        ----------
        out_dir : string
             a string like './models' or '/home/username/models'

        Returns
        -------
        path : string
            model saved path.

        """
        if not hasattr(self, 'saver'):
            assert hasattr(self, 'sess'), 'new a session first.'
            assert hasattr(self, 'model'), 'new a model first.'

            checkpoint_prefix = os.path.join(out_dir, "model")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            assert os.path.exists(out_dir), 'can\'t find the out directory.'

            self.checkpoint_prefix = checkpoint_prefix
            self.saver = tf.train.Saver()

            assert self.model.model_config, ' the model didn\'t init model config and can\'t be saved.!= ='
            del self.model.model_config['self']
            self.model.model_config['class_name'] = self.model.__class__.__name__
            json.dump(self.model.model_config, open(self.checkpoint_prefix + '_config.json', 'w'))

        current_step = tf.train.global_step(self.sess, self.global_step)
        return self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)

    def load(self, model_dir, global_step=None):
        """restore model weights from model directory.

        This method load model weights from model directory and restore
        the model.

        Parameters
        ----------
        model_dir : string
            a string like './models' or '/home/username/models'

        global_step : integer
            a integer of global step
        """
        assert os.path.exists(model_dir), 'can\'t find the model directory.'
        model_config_dir = os.path.join(model_dir, 'model_config.json')
        assert os.path.exists(model_config_dir), 'can\'t find the model_config.json'
        self.model_config_dir = model_config_dir
        self.model_config = json.load(open(self.model_config_dir))

        from .neural_network import deserialize
        model = deserialize(self.model_config['class_name'])

        del self.model_config['class_name']
        self.model = model(**self.model_config)
        self.saver = tf.train.Saver()

        self.checkpoint_prefix = os.path.join(model_dir, "model")
        if global_step:
            self.saver.restore(self.sess, self.checkpoint_prefix + '-' + str(global_step))
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    def __init_summary(self, out_dir, summaries_dict):
        assert hasattr(self, 'sess'), 'new a model first.'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        assert os.path.exists(out_dir), 'can\'t find the out directory.'

        self.summary_writer = tf.summary.FileWriter(out_dir)
        self.summary_writer.add_graph(self.sess.graph)  # init graph first

        if summaries_dict is not None:
            for tag, value in summaries_dict.items():
                if len(value.get_shape()) <= 1:
                    tf.summary.scalar(tag, tf.squeeze(value))
                else:
                    tf.summary.histogram(tag, value)
        self.merged_summaries = tf.summary.merge_all()

    def summaries(self, out_dir, feed_dict, summaries_dict=None):
        """restore model weights from model directory.

        This method load model weights from model directory and restore
        the model.

        Parameters
        ----------
        out_dir : string
            a string like './models' or '/home/username/models'

        feed_dict : dictionary
            a feed dictionary in training process

        summaries_dict : dictionary
            a dictionary of variables should be summarized.

        """
        if not hasattr(self, 'summary_writer'):
            self.__init_summary(out_dir, summaries_dict)
        merged_summary = self.sess.run(self.merged_summaries, feed_dict=feed_dict)
        current_step = tf.train.global_step(self.sess, self.global_step)
        self.summary_writer.add_summary(merged_summary, current_step)

