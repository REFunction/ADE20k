import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
import numpy as np

def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[1]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init,
                          shape=weights.shape)
    return var

def upscore_layer(bottom, shape,
                   num_classes, name, debug,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, num_classes, in_features]

        # create
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input) ** 0.5

        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        if debug:
            deconv = tf.Print(deconv, [tf.shape(deconv)],
                              message='Shape of %s' % name,
                              summarize=4, first_n=1)

    return deconv