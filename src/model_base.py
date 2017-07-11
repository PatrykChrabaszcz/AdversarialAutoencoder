import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, conv2d_transpose, batch_norm


class ModelBase:
    def __init__(self, z_dim, y_dim, non_lin, batch_size):
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.non_lin = non_lin
        self.batch_size = batch_size

        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def discriminator(self, c_i, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            c_i = self._fc(c_i, 500, activation=self.non_lin, bias=True)
            c_i = self._fc(c_i, 500, activation=self.non_lin, bias=True)
            c_i = self._fc(c_i, 1, bias=True)

        return c_i

    def sampler(self):
        z = tf.truncated_normal([self.batch_size, self.z_dim + self.y_dim], name='sampled_z')
        return z

    def _bn(self, c_i, activation=True):
        non_lin = self.non_lin if activation else None
        return batch_norm(c_i, activation_fn=non_lin, is_training=self.is_training, fused=True,
                          data_format="NCHW", updates_collections=None)

    def _conv(self, c_i, fn, kernel_size, stride, activation=None, bias=False):
        b_i = tf.zeros_initializer() if bias else None
        return conv2d(c_i, fn, kernel_size, stride, data_format="NCHW",
                      activation_fn=activation, biases_initializer=b_i)

    def _fc(self, c_i, fn, activation=None, bias=False):
        b_i = tf.zeros_initializer() if bias else None
        return fully_connected(c_i, fn, activation_fn=activation, biases_initializer=b_i)

    def _convt(self, c_i, fn, kernel_size, stride, activation=None, bias=False):
        b_i = tf.zeros_initializer() if bias else None
        return conv2d_transpose(c_i, fn, kernel_size, stride, data_format="NCHW",
                                activation_fn=activation, biases_initializer=b_i)


