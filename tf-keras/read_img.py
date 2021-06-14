import cv2
from PIL import Image, ImageOps
import numpy as np
import os, glob



def read_img_from_directory(from_dir, classes, label_map, type="jpg"):

    def img_example(img, label):

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=2)
        h, w, d = img_array.shape
        tmp = np.zeros(shape=(classes, ), dtype=np.int32)
        tmp[label_map[label]] = 1
        feature = {
            'height' : h,
            'width' : w,
            'depth' : d,
            'img_array' : img_array,
            'label' : tmp
        }

        return feature

    dataset = []
    labels = []
    for root, dirs, files in os.walk(from_dir, topdown=False):
        for dirname in dirs:
            label = dirname
            if len(label) != 1:
                continue
            for imagepath in glob.glob(os.path.join(root, dirname, f'*.{type}')):
                img = Image.open(imagepath)
                img = ImageOps.grayscale(img)
                example = img_example(img, label)
                if example['img_array'].shape == (200, 200, 1):
                    dataset.append(example['img_array'])
                    labels.append(example['label'])
    dataset = np.stack(dataset, axis=0)
    labels = np.stack(labels, axis=0)
    return dataset, labels