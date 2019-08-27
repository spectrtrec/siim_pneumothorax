import keras
import keras.callbacks as callbacks
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
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
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from efficientnet import EfficientNetB4
from keras.engine.training import Model
from keras.layers import SpatialDropout2D, Conv2D
from segmentation_models import Unet


class SnapshotCallbackBuilder:
    def __init__(
        self,
        swa,
        nb_epochs,
        nb_snapshots,
        fold,
        reduce_lr_factor=0.25,
        reduce_lr_patience=10,
        reduce_lr_min=0.00000625,
        init_lr=0.0001,
    ):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.swa = swa
        self.fold = fold
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_min = reduce_lr_min

    def get_callbacks(self, model_prefix="Model"):
        reduce_lr = ReduceLROnPlateau(
            monitor="val_my_iou_metric",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.reduce_lr_min,
            verbose=1,
            mode="max",
        )
        callback_list = [
            callbacks.ModelCheckpoint(
                f"models/keras_{self.fold}.model",
                monitor="val_my_iou_metric",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            self.swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
        ]
        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x


class SWA(keras.callbacks.Callback):
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params["epochs"]
        print(
            "Stochastic weight averaging selected for last {} epochs.".format(
                self.nb_epoch - self.swa_epoch
            )
        )

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (
                    self.swa_weights[i] * (epoch - self.swa_epoch)
                    + self.model.get_weights()[i]
                ) / ((epoch - self.swa_epoch) + 1)

        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print("Final model parameters set to stochastic weight average.")
        self.model.save_weights(self.filepath)
        print("Final stochastic averaged weights saved to file.")


def unet_resnext_50(input_shape, freeze_encoder):
    resnet_base, hyper_list = Unet(
        backbone_name="resnext50",
        input_shape=input_shape,
        input_tensor=None,
        encoder_weights="imagenet",
        freeze_encoder=freeze_encoder,
        skip_connections="default",
        decoder_block_type="transpose",
        decoder_filters=(128, 64, 32, 16, 8),
        decoder_use_batchnorm=True,
        n_upsample_blocks=5,
        upsample_rates=(2, 2, 2, 2, 2),
        classes=1,
        activation="sigmoid",
    )

    x = SpatialDropout2D(0.2)(resnet_base.output)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(x)

    model = Model(resnet_base.input, x)

    return model


def unet_resnext_50_lovasz(input_shape, freeze_encoder):
    resnet_base, hyper_list = Unet(
        backbone_name="resnext50",
        input_shape=input_shape,
        input_tensor=None,
        encoder_weights="imagenet",
        freeze_encoder=freeze_encoder,
        skip_connections="default",
        decoder_block_type="transpose",
        decoder_filters=(128, 64, 32, 16, 8),
        decoder_use_batchnorm=True,
        n_upsample_blocks=5,
        upsample_rates=(2, 2, 2, 2, 2),
        classes=1,
        activation="sigmoid",
    )

    x = SpatialDropout2D(0.2)(resnet_base.output)
    x = Conv2D(1, (1, 1), name="prediction")(x)

    model = Model(resnet_base.input, x)

    return model


def UEfficientNetV2(input_shape=(None, None, 3), dropout_rate=0.1):

    backbone = EfficientNetB4(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    input = backbone.input
    start_neurons = 8

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(
        start_neurons * 32, (3, 3), activation=None, padding="same", name="conv_middle"
    )(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(
        start_neurons * 16, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    deconv4_up1 = Conv2DTranspose(
        start_neurons * 16, (3, 3), strides=(2, 2), padding="same"
    )(deconv4)
    deconv4_up2 = Conv2DTranspose(
        start_neurons * 16, (3, 3), strides=(2, 2), padding="same"
    )(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(
        start_neurons * 16, (3, 3), strides=(2, 2), padding="same"
    )(deconv4_up2)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    #     uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)  # conv1_2

    deconv3 = Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    deconv3_up1 = Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(deconv3)
    deconv3_up2 = Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(deconv3_up1)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, deconv4_up1, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    #     uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    deconv2_up1 = Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(deconv2)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, deconv3_up1, deconv4_up2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    #     uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    #     uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(
        uconv1
    )
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    #     uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    model = Model(input, output_layer)
    model.name = "u-xception"

    return model


def UEfficientNetV1(input_shape=(None, None, 3), dropout_rate=0.1):

    backbone = EfficientNetB4(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(
        start_neurons * 16, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.1)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(
        uconv1
    )
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    model = Model(input, output_layer)
    model.name = "u-xception"

    return model


def get_network(network, input_shape, drop_out):
    if network == "UEfficientNetV2":
        model = UEfficientNetV2(input_shape, drop_out)
        return model
    elif network == "UEfficientNetV1":
        model = UEfficientNetV1(input_shape, drop_out)
        return model
    else:
        raise ValueError("Unknown network " + network)
    return model
