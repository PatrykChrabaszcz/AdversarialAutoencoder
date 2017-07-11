import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer


class NetworkBuilder:
    def __init__(self, data_format="NCHW"):
        self.initializer = xavier_initializer
        self.data_format = data_format

    def get_l2_variable(self, name, shape):
        return tf.get_variable(name,
                               shape=shape,
                               initializer=self.initializer(),
                               dtype=tf.float32,
                               trainable=True,)


    def avg_pool(self, c_i, ksize, stride):
        c_i = tf.nn.avg_pool(c_i, self.strides(ksize), self.strides(stride), 'VALID', self.data_format)
        return c_i

    def slice(self, c_i):
        raise NotImplementedError

    def pad(self, c_i):
        raise NotImplementedError

    def concat(self, list):
        raise NotImplementedError

    def strides(self, stride):
        raise NotImplementedError

    def in_channels(self, c_i):
        raise NotImplementedError

    def width(self, c_i):
        raise NotImplementedError

    def height(self, c_i):
        raise NotImplementedError

    # Reshapes tensor c_i
    def reshape(self, c_i, h, w, c):
        raise NotImplementedError

    # Returns table representing shape
    def get_shape(self, b, h, w, c):
        raise NotImplementedError


class NetworkBuilderNCHW(NetworkBuilder):

    def __init__(self,  bn_settings, lrelu_fac=0, decay=0):
        super(NetworkBuilderNCHW, self).__init__(bn_settings, lrelu_fac, decay, data_format='NCHW')

    def slice(self, c_i):
        return tf.slice(c_i, [0, 0, 1, 1], [-1, -1, -1, -1])

    def pad(self, c_i):
        return tf.pad(c_i, [[0, 0], [0, 0], [0, 1], [0, 1]])

    def concat(self, list):
        try:
            return tf.concat(1, list)
        except:
            return tf.concat(list, 1)

    def strides(self, stride):
        return [1, 1, stride, stride]

    def in_channels(self, c_i):
        return c_i.get_shape()[1]

    def width(self, c_i):
        return c_i.get_shape()[3]

    def height(self, c_i):
        return c_i.get_shape()[2]

    def reshape(self, c_i, h, w, c):
        batch_size = c_i.get_shape()[0]
        return tf.reshape(c_i, [batch_size, c, h, w])

    def get_shape(self, b, h, w, c):
        return [b, c, h, w]


class NetworkBuilderNHWC(NetworkBuilder):
    def __init__(self, bn_settings, lrelu_fac=0, decay=0):
        super(NetworkBuilderNHWC, self).__init__(bn_settings, lrelu_fac, decay, data_format='NHWC')

    def slice(self, c_i):
        return tf.slice(c_i, [0, 1, 1, 0], [-1, -1, -1, -1])

    def pad(self, c_i):
        return tf.pad(c_i, [[0, 0], [0, 1], [0, 1], [0, 0]])

    def concat(self, list):
        try:
            return tf.concat(3, list)
        except:
            return tf.concat(list, 3)

    def strides(self, stride):
        return [1, stride, stride, 1]

    def in_channels(self, c_i):
        return c_i.get_shape()[3]

    def width(self, c_i):
        return c_i.get_shape()[2]

    def height(self, c_i):
        return c_i.get_shape()[1]

    def reshape(self, c_i, h, w, c):
        batch_size = int(c_i.get_shape()[0])
        print(type(batch_size))
        return tf.reshape(c_i, [batch_size, h, w, c])

    def get_shape(self, b, h, w, c):
        return [b, h, w, c]


def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters

