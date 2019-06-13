import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
from backbones import backbone
import numpy as np

def pspnet(input_op, num_classes, is_training, backbone_name='resnet101'):
    # backbone
    net, end_points, vars, pretrained_path = \
        backbone(backbone_name, input_op, is_training, print_end_points=True)
    # pyramid
    input_shape = tf.shape(input_op)
    net = pyramid_pooling_module(net, input_shape, is_training) # OS = 8
    # decoder
    net = slim.conv2d(net, 512, 3, stride=1, padding='SAME', biases_initializer=None, activation_fn=None)
    net = slim.batch_norm(net, is_training=is_training)
    net = tf.nn.relu(net)
    net = slim.dropout(net, keep_prob=0.9, is_training=is_training)
    net = slim.conv2d(net, num_classes, 1, stride=1, activation_fn=None)
    net = tf.image.resize_bilinear(net, [input_shape[1], input_shape[2]])

    return net, vars, pretrained_path

def pyramid_pooling_module(res, input_shape, is_training):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = [tf.shape(res)[1], tf.shape(res)[2]]

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape, is_training)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape, is_training)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape, is_training)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape, is_training)
    res = tf.concat([res, interp_block6, interp_block3, interp_block2, interp_block1], axis=3)
    return res

def interp_block(prev_layer, level, feature_map_shape, input_shape, is_training):
    kernel_strides_map = {1: 90, 2: 45, 3: 30, 6: 15}
    prev_layer = slim.pool(prev_layer, kernel_strides_map[level], padding='SAME',
                           pooling_type='AVG', stride=kernel_strides_map[level])
    prev_layer = slim.conv2d(prev_layer, 256, 1, stride=1, biases_initializer=None, activation_fn=None)
    prev_layer = slim.batch_norm(prev_layer, is_training=is_training)
    prev_layer = tf.nn.relu(prev_layer)
    prev_layer = tf.image.resize_bilinear(prev_layer, size=feature_map_shape)
    return prev_layer
