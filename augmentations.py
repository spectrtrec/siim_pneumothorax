import cv2
from albumentations import (CLAHE, Blur, CenterCrop, Compose, ElasticTransform,
                            GaussNoise, GridDistortion, HorizontalFlip,
                            HueSaturationValue, IAAAdditiveGaussianNoise,
                            JpegCompression, MedianBlur, MotionBlur, OneOf,
                            OpticalDistortion, RandomBrightness,
                            RandomContrast, RandomGamma, RandomSizedCrop,
                            RGBShift, ShiftScaleRotate, ToFloat)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
    ),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    ToFloat(max_value=1)
], p=1)


AUGMENTATIONS_TEST = Compose([ToFloat(max_value=1)], p=1)
