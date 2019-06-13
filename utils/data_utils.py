import numpy as np
import random
import cv2
from math import ceil

def random_crop_with_size(image_batch, label_batch, crop_height=64, crop_width=64):
    image_batch = np.array(image_batch)
    label_batch = np.array(label_batch)

    shape = np.shape(image_batch)

    height = shape[1]
    width = shape[2]

    assert len(shape) == 4
    assert height >= crop_height
    assert width >= crop_width

    height_start = random.randint(0, height - crop_height)
    height_end = height_start + crop_height

    width_start = random.randint(0, width - crop_width)
    width_end = width_start + crop_width

    return image_batch[:, height_start:height_end, width_start: width_end, :], \
           label_batch[:, height_start:height_end, width_start: width_end]
def random_crop_with_max_size(image_batch, label_batch, max_height=64, max_width=64):
    shape = np.shape(image_batch)
    height = shape[1]
    width = shape[2]

    if height > max_height or width > max_width:
        return random_crop_with_size(image_batch, label_batch,
                crop_height=min(height, max_height), crop_width=min(width, max_width))
    else:
        return image_batch, label_batch
def random_crop_anyway_with_size(image_batch, label_batch, crop_height=64, crop_width=64):
    shape = np.shape(image_batch)
    height = shape[1]
    width = shape[2]

    if height < crop_height or width < crop_width:
        rate = max(crop_height / height, crop_width / width)
        image_batch, label_batch = resize_with_rate(image_batch, label_batch, rate)
    return random_crop_with_size(image_batch, label_batch, crop_height, crop_width)

def resize_with_rate(image_batch, label_batch, rate):
    shape = np.shape(image_batch)
    height = shape[1]
    width = shape[2]
    batch_size = shape[0]

    new_height = ceil(height * rate)
    new_width = ceil(width * rate)

    image_batch_resize = []
    label_batch_resize = []

    for i in range(batch_size):
        image_batch_resize.append(cv2.resize(image_batch[i], (new_width, new_height)))
        label_batch_resize.append(cv2.resize(label_batch[i], (new_width, new_height),
                                             interpolation=cv2.INTER_NEAREST))

    return image_batch_resize, label_batch_resize