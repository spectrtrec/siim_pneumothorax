import argparse
import os
import sys

import pandas as pd
from keras import backend as K
from keras_radam import RAdam
from augmentations import *
from losses import *
from model import *
from siim_data_loader import *
from utils import *
from segmentation_models import Unet

parser = argparse.ArgumentParser()
parser.add_argument("--network", default="UEfficientNetV2")
parser.add_argument("--fine_size", default=512, type=int, help="Resized image size")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size for training")
parser.add_argument("--train_path", default="pneumotorax512/train/", help="train path")
parser.add_argument("--masks_path", default="pneumotorax512/masks/", help="mask path")
parser.add_argument("--test_path", default="pneumotorax512/test/", help="test path")
parser.add_argument("--pretrain_weights", help="pretrain weights")
parser.add_argument("--epoch", default=30, type=int, help="Number of training epochs")
parser.add_argument("--swa_epoch", default=15, type=int, help="Number of swa epochs")
parser.add_argument("--debug", default=False, type=bool, help="Debug")
args = parser.parse_args()


def train(
    list_train,
    list_valid,
    train_path,
    masks_path,
    model,
    epoch,
    batch_size,
    fold,
    imh_size,
    swa_epoch,
):
    swa = SWA(f"models/keras_swa_{fold}.model", swa_epoch)
    snapshot = SnapshotCallbackBuilder(swa, epoch, 1, fold)
    training_generator = DataGenerator(
        list_train, train_path, masks_path, AUGMENTATIONS_TRAIN, batch_size, imh_size
    )
    validation_generator = DataGenerator(
        list_valid, train_path, masks_path, AUGMENTATIONS_TEST, batch_size, imh_size
    )

    history = model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False,
        epochs=epoch,
        verbose=2,
        callbacks=snapshot.get_callbacks(),
    )


if __name__ == "__main__":
    debug = args.debug
    df = pd.read_csv("train_proc_v2_gr.csv")
    test_files = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(os.path.join(os.getcwd(), args.test_path))
    ]
    df_test = pd.DataFrame(test_files, columns=["id"])
    epoch = args.epoch
    list_thresh = []
    for fold in [0]:
        print("-----fold-----")
        df_train = df[df.fold != fold].copy().reset_index(drop=True)
        df_valid = df[df.fold == fold].copy().reset_index(drop=True)

        if debug:
            df_train = df[df.fold != 0].copy().reset_index(drop=True)
            df_valid = df[df.fold == 0].copy().reset_index(drop=True)
            df_train = df_train.iloc[:60]
            df_valid = df_train.iloc[:60]
            df_test = df_test.iloc[:60]
            epoch = 3
        K.clear_session()
        model = get_network(
            args.network, input_shape=(args.fine_size, args.fine_size, 3), drop_out=0.5
        )
        model.compile(loss=bce_dice_loss, optimizer="adam", metrics=[my_iou_metric])
        train(
            df_train["id"].values,
            df_valid["id"].values,
            args.train_path,
            args.masks_path,
            model, 
            epoch,
            args.batch_size,
            fold,
            args.fine_size,
            args.swa_epoch,
        )
        try:
            print("using swa weight model")
            model.load_weights(f"models/keras_swa_{fold}.model")
        except Exception as e:
            print(e)
            model.load_weights(f"models/keras_{fold}.model")

        val_predict = predict_validation_result(
            model,
            args.train_path,
            args.masks_path,
            df_valid["id"].values,
            args.batch_size,
            args.fine_size,
        )
        best_threshhold = prderict_best_threshhold(
            df_valid["id"].values, args.masks_path, val_predict, args.fine_size
        )
        list_thresh.append(best_threshhold)
        predict = predict_result(
            model,
            df_test["id"].values,
            args.test_path,
            args.fine_size,
            best_threshhold,
            args.batch_size,
            fold,
        )
        if fold == 0:
            preds_test = predict
        else:
            preds_test += predict
    submit(preds_test, df_test["id"].values, args.network, max(list_thresh))
