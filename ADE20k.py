import numpy as np
import cv2
import os
import queue
import threading
import random
from utils.data_utils import random_crop_anyway_with_size

class ADE20k:
    def __init__(self, root_path='./ADEChallengeData2016/', mode='train', crop_height=0, crop_width=0,
                 flip=False, blur=False):
        self.root_path = root_path
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.flip = flip
        self.blur = blur

        if mode == 'train':
            self.image_dic_path = root_path + 'images/training/'
            self.label_dic_path = root_path + 'annotations/training/'
        elif mode == 'val':
            self.image_dic_path = root_path + 'images/validation/'
            self.label_dic_path = root_path + 'annotations/validation/'
        else:
            print('Wrong Mode')
            exit()
        self._read_filenames()
        self._fill_path_queue()
        self._start_read_queue()
    def _read_filenames(self):
        image_names = os.listdir(self.image_dic_path)
        label_names = os.listdir(self.label_dic_path)
        self.image_paths = []
        self.label_paths = []
        for i in range(len(image_names)):
            self.image_paths.append(self.image_dic_path + image_names[i])
            self.label_paths.append(self.label_dic_path + label_names[i])
    def _fill_path_queue(self):
        if hasattr(self, 'path_queue') == False:
            self.path_queue = queue.Queue()
        for image_path in self.image_paths:
            label_path = self.label_dic_path + image_path.split('/')[-1][:-4] + '.png'
            self.path_queue.put([image_path, label_path])
    def _get_batch(self, batch_size):
        if self.path_queue.qsize() < 30:
            self._fill_path_queue()
        image_batch = []
        label_batch = []
        for i in range(batch_size):
            image_path, label_path = self.path_queue.get()
            image = cv2.imread(image_path)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # label = label - 1 # 151 -> 150

            if self.flip and random.random() > 0.5:
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
            if self.blur and random.random() > 0.8:
                ksize_width = random.randint(0, 3) * 2 + 1
                ksize_height = random.randint(0, 3) * 2 + 1
                sigmaX = random.randint(0, 3)
                image = cv2.GaussianBlur(image, (ksize_width, ksize_height), sigmaX)

            image_batch.append(image)
            label_batch.append(label)

        if self.crop_height != 0:
            image_batch, label_batch = \
                random_crop_anyway_with_size(image_batch, label_batch, self.crop_height, self.crop_width)
        return image_batch, label_batch
    def _add_batch_queue(self):
        while 1:
            image, label = self._get_batch(1)
            self.batch_queue.put([image[0], label[0]])
    def _start_read_queue(self):
        self.batch_queue = queue.Queue(maxsize=30)
        queue_thread = threading.Thread(target=self._add_batch_queue)
        queue_thread.setDaemon(True)
        queue_thread.start()
    def get_batch_fast(self, batch_size):
        image_batch = []
        label_batch = []
        for i in range(batch_size):
            image, label = self.batch_queue.get()
            image_batch.append(image)
            label_batch.append(label)
        return image_batch, label_batch

if __name__ == '__main__':
    ade20k = ADE20k(root_path='h:/ADEChallengeData2016/', mode='train', crop_height=713, crop_width=713,
                    flip=True, blur=True)
    for i in range(1000):
        x, y = ade20k.get_batch_fast(batch_size=4)
        print(np.shape(x), np.shape(y))
        cv2.imshow('image', x[0])
        cv2.imshow('label', y[0])
        cv2.waitKey(0)