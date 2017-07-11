import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, conv2d_transpose, batch_norm
import numpy as np


# k - Increase in the number of features in each convolution layer
# n - Number of convolution repetitions in each resnet block
# n_blocks - Number of resnet blocks. Has to fulfill: n_blocks >= 1
class Model:
    def __init__(self, batch_size, image_size, k=1, n=4, n_blocks=4, non_lin=tf.nn.elu,
                 z_dim=64, y_dim=6):
        assert image_size % 8 == 0
        self.k = k
        self.n = n
        self.n_blocks = n_blocks
        self.non_lin = non_lin

        self.z_dim = z_dim
        self.y_dim = y_dim

        self.rec_lr = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("lr_rec", self.rec_lr)

        self.batch_size = batch_size
        self.image_size = image_size

        self.is_training = tf.placeholder(tf.bool, name='is_training')

    # Encoder extracts images and labels from queue
    # Next multiple convolutions and nonlinearities are applied to
    # the image producing latent representation
    def encoder(self, queue, reuse=False):
        print(queue[0].get_shape())
        c_i = queue[0]
        label = tf.cast(queue[1], dtype=tf.float32)
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()
            c_i = self._conv(c_i, 16, 3, 1)
            c_i = self._bn(c_i)

            # First resnet block
            for _ in range(self.n):
                #with tf.variable_scope('BigBlock') as scope:
                c_i = self._block_conv(c_i, 16*self.k, reduce=False)

            # Additional resnet blocks
            for i in range(1, self.n_blocks):
                c_i = self._block_conv(c_i, 16*self.k*2**i, reduce=True)
                for __ in range(1, self.n):
                    c_i = self._block_conv(c_i, 16*self.k*2**i, reduce=False)
            print(c_i.get_shape())
            print([-1, int(np.prod(c_i.get_shape()[1:]))])
            c_i = tf.reshape(c_i, [-1, int(np.prod(c_i.get_shape()[1:]))])

            c_i = self._fc(c_i, 1024)
            c_i = self._bn(c_i)
            c_i = self._fc(c_i, self.z_dim, bias=True)

            # Concat labels
            if self.y_dim > 0:
                l_i = self._fc(label, 16)
                l_i = self._bn(l_i)
                l_i = self._fc(l_i, 8)
                l_i = self._bn(l_i)
                l_i = self._fc(l_i, self.y_dim, bias=True)

                c_i = tf.concat([c_i, l_i], 1)
            else:
                c_i = c_i
        return c_i

    def decoder(self, c_i, reuse=False):
        # Number of filters and feature map size after last cnn layer in the encoder network
        enc_fn = int(16*self.k*2**(self.n_blocks-1))
        enc_fs = int(self.image_size/(2**(self.n_blocks-1)))
        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()

            c_i = self._fc(c_i, 1024)
            c_i = self._bn(c_i)
            c_i = self._fc(c_i, enc_fn*(enc_fs**2))
            c_i = self._bn(c_i, False)
            c_i = tf.reshape(c_i, [-1, enc_fn, enc_fs, enc_fs])

            for i in range(self.n_blocks-1):
                for _ in range(self.n-1):
                    c_i = self._block_tconv(c_i, enc_fn//(2**i))
                c_i = self._block_tconv(c_i, enc_fn//(2**i), upscale=True)

            # Last block
            for _ in range(self.n):
                c_i = self._block_tconv(c_i, enc_fn//(2**(self.n_blocks-1)))

            c_i = self._convt(c_i, 3, 3, 1, bias=True)
        x_rec = tf.nn.tanh(c_i)
        return x_rec

    def gan(self, c_i, reuse=False):
        with tf.variable_scope("gan") as scope:
            if reuse:
                scope.reuse_variables()

            c_i = self._convt(c_i, 64, 3, 1)
            c_i = self._bn(c_i)
            c_i = self._convt(c_i, 3, 3, 1, bias=True)
        
        x = tf.nn.tanh(c_i)
        return x

    def _block_conv(self, c_i, fn, reduce):
        l_i = c_i
        stride = 2 if reduce else 1
        #with tf.variable_scope('SmallBlock'):
        c_i = self.non_lin(c_i)
        c_i = self._conv(c_i, fn, 3, stride)
        c_i = self._bn(c_i)
        c_i = self._conv(c_i, fn, 3, 1)
        c_i = self._bn(c_i, False)
        # Add input
        # Skip connection from:
        # https://github.com/xgastaldi/shake-shake
        if reduce:
            #with tf.variable_scope('ReduceBlock'):
                # Should I keep this nonlinearity?????????
            l_i = self.non_lin(l_i)
            # Features from upper left corner of 2x2 stride
            l_i_a = tf.nn.avg_pool(l_i, [1, 1, 1, 1], [1, 1, stride, stride], padding="SAME", data_format="NCHW")
            l_i_a = self._conv(l_i_a, fn//2, 1, 1)
            # Features from bottom right corner of 2x2 stride
            l_i_b = tf.slice(l_i, [0, 0, 1, 1], [-1, -1, -1, -1])
            l_i_b = tf.pad(l_i_b, [[0, 0], [0, 0], [0, 1], [0, 1]])
            l_i_b = tf.nn.avg_pool(l_i_b, [1, 1, 1, 1], [1, 1, stride, stride], padding="SAME", data_format="NCHW")
            l_i_b = self._conv(l_i_b, fn//2, 1, 1)
            l_i = tf.concat([l_i_a, l_i_b], 1)
            l_i = self._bn(l_i, False)

        c_i = c_i + l_i
        return c_i

    def _block_tconv(self, c_i, fn, upscale=False):
        l_i = c_i
        stride = 2 if upscale else 1
        fn_2 = fn//2 if upscale else fn
        #with tf.variable_scope('SmallBlockT'):
        c_i = self.non_lin(c_i)
        c_i = self._convt(c_i, fn, 3, 1)
        c_i = self._bn(c_i)
        c_i = self._convt(c_i, fn_2, 3, stride)
        c_i = self._bn(c_i)

        if upscale:
            # Maybe this non lin should not be here
            l_i = self.non_lin(l_i)
            l_i = self._convt(l_i, fn_2, 3, stride)
            l_i = self._bn(l_i, False)
        c_i = c_i + l_i
        return c_i

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
