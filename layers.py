import tensorflow as tf


def linear(x, units, activation=tf.nn.elu, name=None, bn=True):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [x.get_shape().as_list()[-1], units], dtype=tf.float32,
                            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable("b", [units], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        output = tf.nn.xw_plus_b(x, w, b)
        if bn:
            output = tf.layers.batch_normalization(output)
        output = activation(output)
        return output


def deconv_layer(x, filters, kernel, strides, padding, name, activation=tf.nn.relu):
    with tf.variable_scope(name):
        output = tf.layers.conv2d_transpose(x, filters, kernel, strides, padding)
        output = tf.layers.batch_normalization(output)
        output = activation(output)
        return output


def conv_layer(x, filters, kernel, strides, padding, name, activation=tf.nn.elu):
    with tf.variable_scope(name):
        output = tf.layers.conv2d(x, filters, kernel, strides, padding)
        output = tf.layers.batch_normalization(output)
        output = activation(output)
        return output
