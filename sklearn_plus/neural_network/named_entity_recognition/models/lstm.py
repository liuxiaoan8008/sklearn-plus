"""X plus b"""

# Author: Xiaoan Liu <f13221698@gmail.com>

import tensorflow as tf

class LSTM(object):
    def __init__(self, max_sequence_length, vocab_size, label_vocab_size, embedding_size):
        # Note: must add 'self.model_config = locals()' on the top of __init__ function.
        # It is used for model hyparameters save and load.
        self.model_config = locals()

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int64, [None, max_sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, max_sequence_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding_layer"):
            word_vectors = tf.contrib.layers.embed_sequence(
                self.input_x, vocab_size=vocab_size, embed_dim=embedding_size, scope="words")

        with tf.name_scope("lstm_layer"):
            cell = tf.contrib.rnn.LSTMCell(embedding_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            output, _ = tf.nn.dynamic_rnn(cell, word_vectors, dtype=tf.float32)

        with tf.name_scope("fn_layer"):
            self.logits = tf.contrib.layers.fully_connected(output, label_vocab_size)

        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(tf.nn.softmax(self.logits), 2, name="predictions")

        with tf.name_scope("loss"):
            zeros_with_shape = tf.zeros_like(self.input_y, dtype=tf.int64)
            weights = tf.to_double(tf.not_equal(zeros_with_shape, self.input_y))

            target = tf.one_hot(self.input_y, label_vocab_size, 1, 0)
            self.loss = tf.contrib.losses.softmax_cross_entropy(self.logits, target, weights=weights)

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"), name="accuracy")

        print("LOADED LSTM!")
