import sys
import os
import pandas as pd
from masks import *
import cv2
import zipfile

TRAIN = os.path.join(os.getcwd(),  "siim-png-images/train_png/")
TEST = os.path.join(os.getcwd(), "siim-png-images/test_png/")
MASK = os.path.join(os.getcwd(), "train-rle.csv")
mask_rle_df = pd.read_csv(MASK)
mask_rle_df = mask_rle_df.rename(columns={' EncodedPixels': 'EncodedPixels'})
mask_rle_df.info()
img_mask_df = mask_rle_df[mask_rle_df['EncodedPixels'] != ' -1'].groupby('ImageId').count()
img_mask_df = img_mask_df.rename(columns={'EncodedPixels': 'NMasks'})
img_mask_df['EncodedMasks']=''
img_mask_df.head()
i = 0
for index, row in img_mask_df.iterrows():
    #print(imageid)
    rles = mask_rle_df.loc[mask_rle_df['ImageId']==index, 'EncodedPixels'].values
    #print(rles)
    img_mask_df.at[index,'EncodedMasks'] = rles
    i = i+1
print('Total image: ', i)
img_nomask_df = mask_rle_df[mask_rle_df['EncodedPixels'] == ' -1'].groupby('ImageId').sum()
img_nomask_df.head()
OUTPUT = os.path.join(os.getcwd(),  "pneumotorax512/masks/")
if not os.path.exists(OUTPUT):
    os.mkdir(OUTPUT)
out_size = (512,512)
i = 0
j = 0
with zipfile.ZipFile(os.path.join(OUTPUT, 'mask.zip'), 'w') as zip:
    for index, row in img_mask_df.iterrows():
        rles = row['EncodedMasks']
        mask = np.zeros((1024,1024))
        for rle in rles:
            mask = mask + rle2mask(rle, 1024,1024).T
        file = index+'.png'
        _, png = cv2.imencode('.png', cv2.resize(mask, out_size))
        zip.writestr(file, png)
        i = i + 1
    for index, row in img_nomask_df.iterrows():
        mask = np.zeros((1024,1024))
        file = index+'.png'
        _, png = cv2.imencode('.png', cv2.resize(mask, out_size))
        zip.writestr(file, png)
        j = j + 1
print('Total mask files written: ', i+j)


# files_train = [
#     os.path.splitext(filename)[0]
#     for filename in os.listdir(os.path.join(os.getcwd(), "siim-png-images/train_png/"))
# ]
# df_train = pd.DataFrame(files_train, columns=["id"])

# files_mask = [
#     os.path.splitext(filename)[0]
#     for filename in os.listdir(os.path.join(os.getcwd(), "siim-png-images/masks_png/"))
# ]
# df_masks = pd.DataFrame(files_mask, columns=["id"])
# diff = np.setdiff1d(df_train["id"].values, df_masks["id"].values)
# for files in diff:
#     os.remove(os.path.join(os.getcwd(), "siim-png-images/train_png/" + files + ".png"))
# print(diff)
# from PIL import Image

# TRAIN = os.path.join(os.getcwd(), "siim-png-images/test_png/")
# files_train = [
#     os.path.splitext(filename)[0]
#     for filename in os.listdir(os.path.join(os.getcwd(), "siim-png-images/test_png/"))
# ]
# df_train = pd.DataFrame(files_train, columns=["id"])
# #df_train = df_train.iloc[:30]
# image_path = os.path.join(os.getcwd(), "pneumotorax512/")
# for i, image_id in enumerate(df_train["id"].values):
#     im = Image.open(
#         os.path.join(os.getcwd(), "siim-png-images/test_png/") + image_id + ".png"
#     )
#     im = im.resize((512, 512))
#     im.save(os.path.join(os.getcwd(), "pneumotorax512/test/") + image_id + ".png")

