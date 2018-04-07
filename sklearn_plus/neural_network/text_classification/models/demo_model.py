import tensorflow as tf


class Demo(object):
    """docstring for Demo"""

    def __init__(self):
        # input and output
        self.x_data = tf.placeholder(shape=[1], dtype=tf.float32, name="input_x")
        self.y_target = tf.placeholder(shape=[1], dtype=tf.float32, name="input_y")

        # weights
        self.A = tf.get_variable('weight', initializer=tf.random_normal(shape=[1]))

        self.scores = tf.add(self.x_data, self.A, name="scores")

        # probability
        self.logits = tf.nn.tanh(self.scores, name="logits")

        # predictions
        self.predictions = tf.sign(self.logits, name="prediction")

        # loss
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.y_target, name="loss")

        print "loaded demo graph! :)"

