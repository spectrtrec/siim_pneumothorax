import sys
import os
import pandas as pd
from masks import *
import cv2
import zipfile
from PIL import Image


class ResizeImg(object):

    def __init__(
        self, out_img_size, train_path, test_path
    ):
        self.out_size = out_img_size
        self.train = os.path.join(os.getcwd(), train_path)
        self.test = os.path.join(os.getcwd(), test_path)
        self.masks = os.path.join(os.getcwd(), "train-rle.csv")
        self.mask_output = os.path.join(os.getcwd(), mask_out)
        self.train_output = os.path.join(os.getcwd(), train_out)
        self.test_output = os.path.join(os.getcwd(), test_out)

    def resize_masks(self):
        mask_rle_df = pd.read_csv(self.masks)
        mask_rle_df.rename(columns={" EncodedPixels": "EncodedPixels"}, inplace=True)
        img_mask_df = (
            mask_rle_df[mask_rle_df["EncodedPixels"] != " -1"]
            .groupby("ImageId")
            .count()
        )
        img_mask_df.rename(columns={"EncodedPixels": "NMasks"}, inplace=True)
        img_mask_df["EncodedMasks"] = ""
        i = 0
        for index, row in img_mask_df.iterrows():
            rles = mask_rle_df.loc[
                mask_rle_df["ImageId"] == index, "EncodedPixels"
            ].values
            img_mask_df.at[index, "EncodedMasks"] = rles
            i = i + 1
        img_nomask_df = (
            mask_rle_df[mask_rle_df["EncodedPixels"] == " -1"].groupby("ImageId").sum()
        )
        self.save_zip_masks(img_mask_df)

    def save_zip_masks(self, img_mask_df):
        out_size = (self.out_size, self.out_size)
        i = 0
        j = 0
        with zipfile.ZipFile(os.path.join(OUTPUT, "mask.zip"), "w") as zip:
            for index, row in img_mask_df.iterrows():
                rles = row["EncodedMasks"]
                mask = np.zeros((1024, 1024))
                for rle in rles:
                    mask = mask + rle2mask(rle, 1024, 1024).T
                file = index + ".png"
                _, png = cv2.imencode(".png", cv2.resize(mask, out_size))
                zip.writestr(file, png)
                i = i + 1
            for index, row in img_nomask_df.iterrows():
                mask = np.zeros((1024, 1024))
                file = index + ".png"
                _, png = cv2.imencode(".png", cv2.resize(mask, out_size))
                zip.writestr(file, png)
                j = j + 1

    def resize_img(self):
        out_path = [self.train_output, self.train_output]
        for i, path in enumerate([self.train, self.test]):
            files_train = [
                os.path.splitext(filename)[0]
                for filename in os.listdir(
                    os.path.join(os.getcwd(), path)
                )
            ]
            df_train = pd.DataFrame(files_train, columns=["id"])
            for _, image_id in enumerate(df_train["id"].values):
                im = Image.open(
                    os.path.join(os.getcwd(), path)
                    + image_id
                    + ".png"
                )
                im = im.resize((self.out_size, self.out_size))
                im.save(
                    os.path.join(os.getcwd(), out_path[i]) + image_id + ".png"
                )
