"""X plus b"""

# Author: Xiaoan Liu <f13221698@gmail.com>

import tensorflow as tf

class BiLSTMATTCrf(object):
    def __init__(self, max_sequence_length, vocab_size, label_vocab_size, embedding_size):
        # Note: must add 'self.model_config = locals()' on the top of __init__ function.
        # It is used for model hyparameters save and load.
        self.model_config = locals()

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, max_sequence_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding_layer"):
            word_vectors = tf.contrib.layers.embed_sequence(
                self.input_x, vocab_size=vocab_size, embed_dim=embedding_size, scope="words")

        with tf.name_scope("bilstm_layer"):
            fw_cell = tf.contrib.rnn.LSTMCell(embedding_size)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.contrib.rnn.LSTMCell(embedding_size)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

            output_raw, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_vectors, dtype=tf.float32)
            output = tf.concat(output_raw, 2)

        with tf.name_scope("attention"):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(embedding_size, output)
            fw_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
                fw_cell, attention_mechanism, attention_size=embedding_size)
            bw_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
                bw_cell, attention_mechanism, attention_size=embedding_size)


        with tf.name_scope("fn_layer"):
            self.logits = tf.contrib.layers.fully_connected(output, label_vocab_size)
            self.logits = tf.reshape(self.logits, [-1, max_sequence_length, label_vocab_size])

        with tf.name_scope("crf_layer"):
            self.sequence_lengths = tf.count_nonzero(self.input_y, axis=1)
            log_likelihood, self.transition_param = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.input_y, self.sequence_lengths)


        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(-log_likelihood)

        print("LOADED BILSTM-ATT-CRF!")
