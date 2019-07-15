import gc
import glob
import os
import shutil

import cv2
import keras
import keras.backend as K
import keras.callbacks as callbacks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras import constraints, initializers, optimizers, regularizers
from keras.applications.xception import Xception
from keras.callbacks import Callback, ModelCheckpoint
from keras.engine import InputSpec
from keras.engine.topology import Input, get_source_inputs
from keras.engine.training import Model
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    LeakyReLU,
    MaxPooling2D,
    Permute,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
    multiply,
)
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.core import Activation, Dense, Lambda, SpatialDropout2D
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.legacy import interfaces
from keras.losses import binary_crossentropy
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.regularizers import l2
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.utils.generic_utils import get_custom_objects
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm_notebook


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        list_ids,
        image_path,
        mask_path,
        augmentations,
        batch_size,
        img_size,
        n_channels=3,
        shuffle=True,
    ):
        self.indexes = []
        self.image_path = image_path
        self.masks_path = mask_path
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_ids) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[
            index
            * self.batch_size : min((index + 1) * self.batch_size, len(self.list_ids))
        ]
        list_IDs_im = [self.list_ids[k] for k in indexes]

        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented["image"])
                mask.append(augmented["mask"])
            return np.array(im), np.array(mask) / 255

    def data_generation(self, list_id_index):
        X = np.empty((len(list_id_index), self.img_size,
                      self.img_size, self.n_channels))
        y = np.empty((len(list_id_index), self.img_size, self.img_size, 1))
        for i, image_id in enumerate(list_id_index):
            im = np.array(Image.open(os.path.join(
                os.getcwd(), self.image_path) + image_id + '.png'))
            mask = np.array(Image.open(os.path.join(
                os.getcwd(), self.masks_path) + image_id + '.png'))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)
            X[i, ] = cv2.resize(im, (self.img_size, self.img_size))

            y[i, ] = cv2.resize(
                mask, (self.img_size, self.img_size))[..., np.newaxis]
            y[y > 0] = 255

        return np.uint8(X), np.uint8(y)
