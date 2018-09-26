"""X plus b"""

# Author: Xiaoan Liu <f13221698@gmail.com>

import tensorflow as tf


class XPlusB(object):
    """A 'x plus b' model for contributors to read and try.

    Problem Definition:

    I faked data like this:
        x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
        y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

    Then I defined a model to do binary classification:
        y = x + b
        prob = sigmoid(y)
        loss = cross_entropy(prob)
        b: the weight to learn.

    Notes:
        1. Must add 'self.model_config = locals()' on the top of __init__ function.
        It is used for model hyparameters save and load.

        2. Some tensors we care in training and predicting, please follow the rules
        below:
            * use 'self.input_x' to represent for input x tensor.
            * use 'self.input_y' to represent for input y tensor.
            * use 'self.scores' to represent for raw model output.
            * use 'self.logits' to represent for model output probability.
            * use 'self.predictions' to represent for model predictions.
            * use 'self.loss' to represent for model loss.

        3. Add 'self.' in front of a tensor which you care and maybe used in
        training and predicting.

    """

    def __init__(self, hyparameters=None):
        # Note: must add 'self.model_config = locals()' on the top of __init__ function.
        # It is used for model hyparameters save and load.
        self.model_config = locals()

        # define model graph here.
        self.input_x = tf.placeholder(shape=[1], dtype=tf.float32, name="input_x")
        self.input_y = tf.placeholder(shape=[1], dtype=tf.float32, name="input_y")

        self.b = tf.get_variable("weights", initializer=tf.random_normal(shape=[1]))

        self.scores = tf.add(self.input_x, self.b, name="scores")
        self.logits = tf.nn.sigmoid(self.scores, name="logits")
        bool_result = tf.less(self.logits, 0.5, name=None)
        self.predictions = tf.cast(bool_result, tf.int32, name="predictions")
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y, name="loss")

        print("loaded 'y = x + b' graph! :)")
