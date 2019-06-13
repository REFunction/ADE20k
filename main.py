import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os
import numpy as np
from net_models.pspnet import pspnet
from net_models.fcn import fcn
from ADE20k import ADE20k


class Model:
    def build(self, num_classes=151, is_training=True):
        print('building network...')
        self.num_classes = num_classes
        self.is_training = is_training
        self.input_op = tf.placeholder(tf.float32, [None, None, None, 3])
        self.label_op = tf.placeholder(tf.int32, [None, None, None])

        net, vars, pretrained_path = \
            pspnet(self.input_op, self.num_classes, is_training=is_training, backbone_name='resnet101')
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

    def train(self, batch_size=4, learning_rate=1e-4, model_path='model/model.ckpt', start_step=0,
              max_steps=1000000, weight_decay=1e-4, pretrain=True, new_train=False,
              data_path='./ADEChallengeData2016/', crop_height=713, crop_width=713,
              print_every=10, save_every=3000, eval_every=1000, flip=False, blur=False):
        # define loss
        self.create_loss_function(weight_decay=weight_decay)
        # learning rate, optimizer and train op
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

        for iter in range(start_step, max_steps):
            start_time = time.time()
            batch_x, batch_y = ade20k.get_batch_fast(batch_size=batch_size)
            feed_dict = {self.input_op: batch_x, self.label_op: batch_y}
            self.sess.run(self.train_op, feed_dict=feed_dict)
            if iter % print_every == 0:
                loss_value, cross_entropy_value = self.sess.run([self.loss, self.cross_entropy], feed_dict=feed_dict)
                print('iter:', iter, 'loss:', round(loss_value, 3),
                      'cross_entropy:', round(cross_entropy_value, 3),
                      'time:', round((time.time() - start_time), 3), 's')
            if iter % save_every == 0 and iter > 0:
                saver.save(self.sess, 'model/model.ckpt')
            if iter % eval_every == 0 and iter > 0:
                miou_value = self.eval(data_path=data_path, restore_path=model_path)

    def eval(self, data_path='./ADEChallengeData2016/', restore_path='model/model.ckpt', eval_num=100):
        # sess
        if hasattr(self, 'sess') == False:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_path)
            print('Restore from', restore_path)
        if hasattr(self, 'miou_op') == False:
            # define ops
            self.miou_op, self.miou_update_op = tf.metrics.mean_iou(num_classes=self.num_classes,
                                                                 labels=tf.reshape(self.label_op, [-1]),
                                                                 predictions=tf.reshape(self.net, [-1]))
            self.local_init_op = tf.local_variables_initializer()
        self.sess.run(self.local_init_op)

        ade20k = ADE20k(root_path=data_path, mode='val')

        for i in range(eval_num):
            batch_x, batch_y = ade20k.get_batch_fast(batch_size=1)
            self.sess.run(self.miou_update_op, feed_dict={self.input_op: batch_x, self.label_op: batch_y})
        miou_value = self.sess.run(self.miou_op)
        print('miou:', miou_value)

        if self.is_training:
            logs_file = open('logs/val_logs.txt', 'a')
            logs_file.write(str(iter) + ' miou:' + str(miou_value) + '\r\n')
            logs_file.close()
        return miou_value

if __name__ == '__main__':
    model = Model()
    model.build(num_classes=151, is_training=True)
    if model.is_training:
        model.train(batch_size=4,
                    learning_rate=1e-5,
                    model_path='model/model.ckpt',
                    start_step=123000,
                    max_steps=1000000,
                    weight_decay=1e-4,
                    pretrain=True,
                    new_train=False,
                    data_path='/media/datasets/ADEChallengeData2016/',
                    crop_height=713,
                    crop_width=713,
                    flip=True,
                    blur=True)
    else:
        model.eval(data_path='/media/datasets/ADEChallengeData2016/', restore_path='model/model.ckpt')