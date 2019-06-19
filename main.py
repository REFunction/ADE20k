import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os
import numpy as np
from net_models.pspnet import pspnet
from net_models.fcn import fcn
from ADE20k import ADE20k
from tensorpack.tfutils.optimizer import AccumGradOptimizer


class Model:
    def build(self, num_classes=151):
        print('building network...')
        self.num_classes = num_classes
        self.input_op = tf.placeholder(tf.float32, [None, None, None, 3])
        self.label_op = tf.placeholder(tf.int32, [None, None, None])
        self.is_training = tf.placeholder(tf.bool)

        net, vars, pretrained_path = \
            pspnet(self.input_op, self.num_classes, is_training=self.is_training, backbone_name='resnet101')
        self.logits = net
        self.vars = vars
        self.pretrained_path = pretrained_path
        self.net = tf.argmax(net, axis=3, output_type=tf.int32)  # [batch_size, height, width]
    def create_loss_function(self, weight_decay=1e-4):
        self.cross_entropy = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                    labels=self.label_op)))
        reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay),
                                                     tf.trainable_variables())
        self.loss = self.cross_entropy + reg

    def train(self, batch_size=4, iter_size=1, learning_rate=1e-4, model_path='model/model.ckpt',
              start_step=0, max_steps=1000000, weight_decay=1e-4, pretrain=True, new_train=False,
              data_path='./ADEChallengeData2016/', crop_height=713, crop_width=713,
              print_every=10, eval_every=1000, flip=False, blur=False):
        # define loss
        self.create_loss_function(weight_decay=weight_decay)
        # learning rate, optimizer and train op
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = AccumGradOptimizer(optimizer, iter_size)
        self.train_op = slim.learning.create_train_op(self.loss, optimizer)
        # sess
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # init or restore
        if new_train:
            self.sess.run(tf.global_variables_initializer())
            if pretrain:
                saver = tf.train.Saver(var_list=self.vars)
                saver.restore(self.sess, self.pretrained_path)
                print('Using pretrained model:', self.pretrained_path)
            print('Initialize all parameters')
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
            print('Restore from', model_path)
        saver = tf.train.Saver()
        # dataset
        ade20k = ADE20k(root_path=data_path, mode='train', crop_height=crop_height, crop_width=crop_width,
                        flip=flip, blur=blur)

        max_miou = 0

        for iter in range(start_step, max_steps):
            start_time = time.time()
            batch_x, batch_y = ade20k.get_batch_fast(batch_size=batch_size)
            feed_dict = {self.input_op: batch_x, self.label_op: batch_y, self.is_training: True}
            self.sess.run(self.train_op, feed_dict=feed_dict)
            if iter % print_every == 0:
                loss_value, cross_entropy_value = self.sess.run([self.loss, self.cross_entropy], feed_dict=feed_dict)
                print('iter:', iter, 'loss:', round(loss_value, 3),
                      'cross_entropy:', round(cross_entropy_value, 3),
                      'time:', round((time.time() - start_time), 3), 's')
            if iter % eval_every == 0 and iter > 0:
                miou_value = self.eval(iter, data_path=data_path, restore_path=model_path)
                if miou_value > max_miou:
                    saver.save(self.sess, 'model/model.ckpt')
                    max_miou = miou_value
                    print('Current Max mIoU:', max_miou)

    def eval(self, iter=0, data_path='./ADEChallengeData2016/', restore_path='model/model.ckpt', eval_num=100):
        # sess
        if hasattr(self, 'sess') == False:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        if hasattr(self, 'miou_op') == False:
            # define ops
            label_1d = tf.reshape(self.label_op, [-1])
            net_1d = tf.reshape(self.net, [-1])
            zero = tf.zeros_like(label_1d)
            ones = tf.ones_like(label_1d)
            weights = tf.where(tf.equal(label_1d, 0), zero, ones) # ignore background
            self.miou_op, self.miou_update_op = tf.metrics.mean_iou(num_classes=self.num_classes,
                                                                 labels=label_1d,
                                                                 predictions=net_1d, weights=weights)
            self.acc_op, self.acc_update_op = tf.metrics.accuracy(labels=label_1d, predictions=net_1d, weights=weights)
            self.local_init_op = tf.local_variables_initializer()
        self.sess.run(self.local_init_op)

        ade20k = ADE20k(root_path=data_path, mode='val')

        for i in range(eval_num):
            batch_x, batch_y = ade20k.get_batch_fast(batch_size=1)
            self.sess.run([self.miou_update_op, self.acc_update_op], feed_dict={self.input_op: batch_x,
                                                          self.label_op: batch_y, self.is_training:False})
        miou_value, acc_value = self.sess.run([self.miou_op, self.acc_op])
        print('miou:', miou_value, 'acc:', acc_value)

        logs_file = open('logs/val_logs.txt', 'a')
        logs_file.write(str(iter) + ' miou:' + str(miou_value) + ' acc:' + acc_value + '\r\n')
        logs_file.close()
        return miou_value

if __name__ == '__main__':
    model = Model()
    model.build(num_classes=151)
    is_training = True
    if is_training:
        model.train(batch_size=8,
                    iter_size=2,
                    learning_rate=1e-5,
                    model_path='model/model.ckpt',
                    start_step=121001,
                    max_steps=1000000,
                    weight_decay=1e-4,
                    pretrain=True,
                    new_train=False,
                    data_path='/media/datasets/ADEChallengeData2016/',
                    crop_height=512,
                    crop_width=512,
                    flip=True,
                    blur=True)
    else:
        model.eval(data_path='/media/datasets/ADEChallengeData2016/', 
                   restore_path='model/model.ckpt',
                   eval_num=100)