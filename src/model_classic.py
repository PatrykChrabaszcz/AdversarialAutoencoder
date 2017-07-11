from src.model_base import ModelBase
import tensorflow as tf
import numpy as np


# k - Increase in the number of features in each convolution layer
class ModelClassic(ModelBase):
    def __init__(self, batch_size, image_size, k=8, steps=4, non_lin=tf.nn.elu,
                 z_dim=64, y_dim=6):
        super().__init__(z_dim, y_dim, non_lin, batch_size)
        assert image_size % 8 == 0
        self.k = k
        self.steps = steps
        self.non_lin = non_lin

        self.image_size = image_size

    def encoder(self, queue, reuse=False):
        c_i = queue[0]
        label = tf.cast(queue[1], dtype=tf.float32)

        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()

            c_i = self._conv(c_i, 16, 3, 1)
            c_i = self._bn(c_i)

            for i in range(self.steps):
                c_i = self._conv(c_i, 16 * self.k * 2**i, 4, 2)
                c_i = self._bn(c_i)

            c_i = tf.reshape(c_i, [-1, int(np.prod(c_i.get_shape()[1:]))])

            c_i = self._fc(c_i, self.z_dim, bias=True)

            # Concat labels
            if self.y_dim > 0:
                l_i = self._fc(label, 16, bias=True, activation=self.non_lin)
                l_i = self._fc(l_i, 8, bias=True, activation=self.non_lin)
                l_i = self._fc(l_i, self.y_dim, bias=True)

                c_i = tf.concat([c_i, l_i], 1)

        return c_i

    def decoder(self, c_i, reuse=False):
        # Number of filters and feature map size after last cnn layer in the encoder network
        enc_fn = int(16 * self.k * 2 ** (self.steps-1))
        enc_fs = int(self.image_size / (2 ** self.steps))
        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()

            c_i = self._fc(c_i,  enc_fn * (enc_fs ** 2), bias=True, activation=self.non_lin)

            c_i = tf.reshape(c_i, [-1, enc_fn, enc_fs, enc_fs])

            for i in range(self.steps):
                c_i = self._convt(c_i, enc_fn // (2 ** i), 4, 2)
                c_i = self._bn(c_i)

            c_i = self._conv(c_i, 3, 3, 1, bias=True)

            x_rec = tf.nn.tanh(c_i)

        return x_rec

    def gan(self, c_i, reuse=False):
        with tf.variable_scope("gan") as scope:
            if reuse:
                scope.reuse_variables()

            c_i = self._conv(c_i, 16*self.k, 3, 1)
            c_i = self._bn(c_i)
            c_i = self._conv(c_i, 16*self.k, 3, 1)
            c_i = self._bn(c_i)
            c_i = self._conv(c_i, 3, 3, 1, bias=True)

        c_i = tf.nn.tanh(c_i)
        return c_i

    def critic(self, c_i, reuse=False):
        with tf.variable_scope("critic") as scope:
            if reuse:
                scope.reuse_variables()

            c_i = self._conv(c_i, 16*self.k, 3, 1)
            c_i = self._bn(c_i)
            c_i = self._conv(c_i, 32*self.k, 3, 2)
            c_i = self._bn(c_i)
            c_i = self._conv(c_i, 64*self.k, 3, 2)
            c_i = self._bn(c_i)
            c_i = self._conv(c_i, 128*self.k, 3, 2)
            c_i = self._bn(c_i)

            c_i = tf.reshape(c_i, [-1, int(np.prod(c_i.get_shape()[1:]))])
            c_i = self._fc(c_i, 1, bias=True)

        return c_i
