from losses import *
from augmentations import *
from model import *
from siim_data_loader import *
from utils import *

import numpy as np
import pandas as pd
import gc
import keras

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, StratifiedKFold

from skimage.transform import resize
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    Dropout,
    BatchNormalization,
)
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from tqdm import tqdm_notebook
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply


from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dropout, BatchNormalization, Activation, Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import concatenate, add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import glob
import shutil
import os
from PIL import Image


def train(list_train, list_valid, model, epoch, batch_size):
    epochs = epoch
    swa = SWA("./keras_swa.model", epoch - 1)
    snapshot = SnapshotCallbackBuilder(
        swa, nb_epochs=epochs, nb_snapshots=1, init_lr=1e-3
    )
    training_generator = DataGenerator(
        list_train,
        "pneumotorax256/train/",
        "pneumotorax256/masks/",
        AUGMENTATIONS_TRAIN,
        batch_size,
        256,
    )
    validation_generator = DataGenerator(
        list_valid,
        "pneumotorax256/train/",
        "pneumotorax256/masks/",
        AUGMENTATIONS_TEST,
        batch_size,
        256,
    )

    history = model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False,
        epochs=epochs,
        verbose=2,
        callbacks=snapshot.get_callbacks(),
    )


if __name__ == "__main__":
    debug = False
    df = pd.read_csv("train_proc_v2_gr.csv")
    test_files = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(os.path.join(os.getcwd(), "pneumotorax256/test"))
    ]
    df_test = pd.DataFrame(test_files, columns=["id"])

    df_train = df[df.fold != 0].copy().reset_index(drop=True)
    df_valid = df[df.fold == 0].copy().reset_index(drop=True)
    epoch = 120
    if debug:
        df_train = df_train.iloc[:60]
        df_valid = df_train.iloc[:60]
        df_test = df_test.iloc[:60]
        epoch = 2
    K.clear_session()
    model = UEfficientNet(input_shape=(256, 256, 3), dropout_rate=0.25)
    model.compile(loss=bce_dice_loss, optimizer="adam", metrics=[my_iou_metric])
    train(df_train["id"].values, df_valid["id"].values, model, epoch, 10)
    try:
        print("using swa weight model")
        model.load_weights("./keras_swa.model")
    except Exception as e:
        print(e)
        model.load_weights("./keras.model")

    val_predict = predict_validation_result(model, df_valid["id"].values, 10, 256)
    best_thresh = prderict_best_threshhold(
        df_valid["id"].values, "pneumotorax256/masks/", val_predict, 256
    )
    predict_result(
        model, df_test["id"].values, "pneumotorax256/test/", 256, best_thresh, 10
    )
