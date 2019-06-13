import tensorflow as tf
import tensorflow.contrib.slim as slim
from backbones import backbone
from math import ceil
import numpy as np

def TestNet(input_op, num_classes, is_training, backbone_name='vgg19'):
    # input_op: [image_op, last_logit_op] [batch_size, height, width, 4]
    # down sample os = 16
    net, end_points, vars, pretrained_path = \
        backbone(backbone_name, input_op, is_training, print_end_points=True)
    # upsample
    net = slim.conv2d_transpose(net, 4096, 4, stride=2)
    net = slim.batch_norm(net, is_training=is_training)

    net = slim.conv2d_transpose(net, 2048, 16, stride=8)
    net = slim.batch_norm(net, is_training=is_training)
    #mask head
    net = slim.conv2d(net, num_classes, 3, activation_fn=None)
    net = tf.image.resize_bilinear(net, [tf.shape(input_op)[1], tf.shape(input_op)[2]])

    return net, vars, pretrained_path