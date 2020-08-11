import argparse
import os

import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--out_size", default=512, type=int, help="Resized image size")
parser.add_argument(
    "--train_path",
    default="pneumotorax1024/train/",
    type=str,
    help="Original train image path",
)
args = parser.parse_args()

if __name__ == "__main__":
    n_fold = 5
    files_train = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(os.path.join(os.getcwd(), args.train_path))
    ]
    df_train = pd.DataFrame(files_train, columns=["id"])
    df_train["fold"] = (list(range(n_fold)) * df_train.shape[0])[: df_train.shape[0]]
    df_train[["id", "fold"]].sample(frac=1, random_state=123).to_csv(
        os.path.join(os.getcwd(), "", "train_proc_v2_gr.csv"), index=False
    )
