import numpy as np
from losses import *
from siim_data_loader import *
from augmentations import *
from masks import *


def predict_validation_result(
    model, path_train, path_mask, list_valid, batch_size, img_size
):
    validation_generator = DataGenerator(
        list_valid,
        path_train,
        path_mask,
        AUGMENTATIONS_TEST,
        batch_size,
        img_size,
        shuffle=False,
    )

    preds_test1 = model.predict_generator(validation_generator).reshape(
        -1, img_size, img_size
    )
    return preds_test1


def predict_result(
    model, test_list, image_path, img_size, threshold_best, batch_size, fold
):
    x_test = [
        np.array(Image.open(os.path.join(os.getcwd(), image_path) + fn + ".png"))
        for fn in test_list
    ]
    x_test = np.array(x_test)
    x_test = np.array([np.repeat(im[..., None], 3, 2) for im in x_test])
    preds_test = model.predict(x_test, batch_size=batch_size)
    return preds_test


def submit(preds_test, test_list, model, threshold_best):
    rles = []
    for p in tqdm_notebook(preds_test):
        p = p.squeeze()
        im = cv2.resize(p, (1024, 1024))
        im = im > threshold_best
        if im.sum() < 1024 * 2:
            im[:] = 0
        im = (im.T * 255).astype(np.uint8)
        rles.append(mask2rle(im, 1024, 1024))

    sub_df = pd.DataFrame({"ImageId": test_list, "EncodedPixels": rles})
    sub_df.loc[sub_df.EncodedPixels == "", "EncodedPixels"] = "-1"
    sub_df.to_csv(
        os.path.join(os.getcwd(), "submissions/", f"submission_{model}.csv"), index=False
    )


def prderict_best_threshhold(validation_list, image_path, preds_valid, img_size):
    y_valid_ori = np.array(
        [
            np.array(Image.open(os.path.join(os.getcwd(), image_path) + fn + ".png"))
            for fn in validation_list
        ]
    )
    thresholds = np.linspace(0.2, 0.9, 31)
    ious = np.array(
        [
            iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold))
            for threshold in tqdm_notebook(thresholds)
        ]
    )
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    return threshold_best
