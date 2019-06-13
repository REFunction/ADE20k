import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

def conv_block(input_op, out_channel, training=True):
    net = slim.conv2d(input_op, out_channel, 3, stride=2, activation_fn=None)
    net = slim.batch_norm(net, is_training=training)
    net = tf.nn.relu(net)
    return net
def spatial_path(input_op, training=True):
    net = conv_block(input_op, 64, training=training)
    net = conv_block(net, 128, training=training)
    net = conv_block(net, 256, training=training)
    return net
def arm(input_op, out_channel):
    net = tf.reduce_mean(input_op, [1, 2], keepdims=True)# global average pooling
    net = slim.conv2d(net, out_channel, 1, activation_fn=None)
    net = tf.sigmoid(net)
    net = tf.multiply(input_op, net)
    return net
def context_path_101(input_op, training=True):

    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope(batch_norm_decay=0.9997)):
        net, end_points = nets.resnet_v2.resnet_v2_101(input_op, num_classes=None,
                                                       is_training=training, output_stride=32, global_pool=False)
    base_model = 'resnet_v2_101'
    exclude = [base_model + '/logits', 'global_step']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

    cx1 = end_points['resnet_v2_101/block3/unit_1/bottleneck_v2']
    cx2 = end_points['resnet_v2_101/block4']
    tail = tf.reduce_mean(cx2, [1, 2], keepdims=True)
    return cx1, cx2, tail, variables_to_restore
def context_path_50(input_op, training=True):

    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope(batch_norm_decay=0.9997)):
        net, end_points = nets.resnet_v2.resnet_v2_50(input_op, num_classes=None,
                                                       is_training=training, output_stride=32, global_pool=False)
    base_model = 'resnet_v2_50'
    exclude = [base_model + '/logits', 'global_step']
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

    cx1 = end_points['resnet_v2_50/block3/unit_1/bottleneck_v2']
    cx2 = end_points['resnet_v2_50/block4']
    tail = tf.reduce_mean(cx2, [1, 2], keepdims=True)
    return cx1, cx2, tail, variables_to_restore
def ffm(input_op1, input_op2, num_classes, training=True):
    input_op1 = tf.image.resize_nearest_neighbor(input_op1,
                                [tf.shape(input_op2)[1], tf.shape(input_op2)[2]])
    net = tf.concat([input_op1, input_op2], axis=3)
    feature = conv_block(net, out_channel=num_classes, training=training)
    net = tf.reduce_mean(feature, [1, 2], keepdims=True)
    net = slim.conv2d(net, num_classes, 1)
    net = slim.conv2d(net, num_classes, 1, activation_fn=tf.sigmoid)
    net = tf.multiply(feature, net)
    net = tf.add(net, feature)
    return net