import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
import numpy as np
from net_models.upsample import upscore_layer
from backbones import backbone


def fcn(input_op, num_classes, is_training, backbone_name='vgg16'):
    net = input_op
    net, end_points, vars, pretrained_path = \
        backbone(backbone_name, input_op, is_training, print_end_points=True)

    base_model = 'vgg_16'
    exclude = [base_model + '/fc6', base_model + '/fc7',
               base_model + '/fc8', base_model + '/logits', 'global_step']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

    pool3 = end_points['vgg_16/pool3']
    pool4 = end_points['vgg_16/pool4']
    pool5 = end_points['vgg_16/pool5']
    conv6 = slim.conv2d(pool5, 4096, 7)
    if is_training:
        conv6 = slim.dropout(conv6)
    conv7 = slim.conv2d(conv6, 4096, 1)
    if is_training:
        conv7 = slim.dropout(conv7)
    conv7 = slim.conv2d(conv7, num_classes, 1)
    # upsampling
    #conv7 = slim.conv2d_transpose(conv7, 21, 4, 2)
    conv7 = upscore_layer(bottom=conv7, shape=None, num_classes=num_classes, name='up_conv7', debug=False)
    pool4 = slim.conv2d(pool4, num_classes, 1)
    pool4 = tf.slice(pool4, [0, 0, 0, 0], [tf.shape(conv7)[0], tf.shape(conv7)[1],
                                           tf.shape(conv7)[2], tf.shape(conv7)[3]])
    conv7_pool4 = conv7 + pool4 * 0.01
    conv7_pool4 = upscore_layer(bottom=conv7_pool4, shape=None, num_classes=num_classes, name='up_conv7_pool4'
                                , debug=False)
    pool3 = slim.conv2d(pool3, num_classes, 1)
    pool3 = tf.slice(pool3, [0, 0, 0, 0], [tf.shape(conv7_pool4)[0], tf.shape(conv7_pool4)[1],
                                           tf.shape(conv7_pool4)[2], tf.shape(conv7_pool4)[3]])
    pool3_pool4_conv7 = pool3 * 0.0001 + conv7_pool4
    net = tf.image.resize_bilinear(pool3_pool4_conv7, [tf.shape(input_op)[1], tf.shape(input_op)[2]])

    return net, vars, pretrained_path