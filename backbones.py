import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def backbone(name, input_op, is_training=True, print_end_points=True):
    if name == 'resnet101':
        with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
            net, end_points = nets.resnet_v2.resnet_v2_101(input_op, num_classes=None,
                                        is_training=is_training, output_stride=16, global_pool=False)
        base_model = 'resnet_v2_101'
        exclude = [base_model + '/logits', 'global_step']
        vars = slim.get_variables_to_restore(exclude=exclude)
        pretrained_path = '/media/backbones/resnet_v2_101.ckpt'
    elif name == 'resnet50':
        with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
            net, end_points = nets.resnet_v2.resnet_v2_50(input_op, num_classes=None,
                                        is_training=is_training, output_stride=16, global_pool=False)
        base_model = 'resnet_v2_50'
        exclude = [base_model + '/logits', 'global_step']
        vars = slim.get_variables_to_restore(exclude=exclude)
        pretrained_path = '/media/backbones/resnet_v2_50.ckpt'
    elif name == 'vgg16':
        with slim.arg_scope(nets.vgg.vgg_arg_scope()):
            net, end_points = nets.vgg.vgg_16(input_op, 512, is_training)
            net = end_points['vgg_16/pool5']
        base_model = 'vgg_16'
        exclude = [base_model + '/fc6', base_model + '/fc7',
                   base_model + '/fc8', base_model + '/logits', 'global_step']
        vars = slim.get_variables_to_restore(exclude=exclude)
        pretrained_path = '/media/backbones/vgg_16.ckpt'
    elif name == 'vgg19':
        with slim.arg_scope(nets.vgg.vgg_arg_scope()):
            net, end_points = nets.vgg.vgg_19(input_op, 512, is_training)
            net = end_points['vgg_19/pool5']
        base_model = 'vgg_19'
        exclude = [base_model + '/fc6', base_model + '/fc7',
                   base_model + '/fc8', base_model + '/logits', 'global_step']
        vars = slim.get_variables_to_restore(exclude=exclude)
        pretrained_path = '/media/backbones/vgg_19.ckpt'

    return net, end_points, vars, pretrained_path

