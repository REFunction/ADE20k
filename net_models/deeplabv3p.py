import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
from math import ceil
import numpy as np
from backbones import backbone

def deeplabv3p(input_op, num_classes, is_training, backbone_name='resnet101'):
    net = input_op
    net, end_points, vars, pretrained_path = \
        backbone(backbone_name, input_op, is_training, print_end_points=True)

    inputs_size = tf.shape(input_op)[1:3]
    net = end_points[base_model + '/block4']
    encoder_output = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

    with tf.contrib.slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with tf.variable_scope("low_level_features"):
                low_level_features = end_points[base_model + '/block1/unit_3/bottleneck_v2/conv1']
                low_level_features = slim.conv2d(low_level_features, 48,
                                                       1, stride=1, scope='conv_1x1')
                low_level_features_size = tf.shape(low_level_features)[1:3]

            with tf.variable_scope("upsampling_logits"):
                net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
                net = tf.concat([net, low_level_features], axis=3, name='concat')
                net = slim.conv2d(net, 256, 3, stride=1, scope='conv_3x3_1')
                net = slim.conv2d(net, 256, 3, stride=1, scope='conv_3x3_2')
                net = slim.conv2d(net, num_classes, 1, activation_fn=None, normalizer_fn=None,
                                        scope='conv_1x1')
                net = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

    return net, vars, pretrained_path

def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.
  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(nets.resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = slim.conv2d(inputs, depth, 1, stride=1, scope="conv_1x1")
        conv_3x3_1 = slim.conv2d(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = slim.conv2d(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = slim.conv2d(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = slim.conv2d(image_level_features, depth, 1, stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = slim.conv2d(net, depth, 1, stride=1, scope='conv_1x1_concat')

        return net