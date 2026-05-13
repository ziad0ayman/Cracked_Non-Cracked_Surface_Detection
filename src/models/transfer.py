import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation
from src.utils.config import get_model, get_global

def build_transfer(**kwargs):
    cfg = {**get_model("transfer"), **kwargs}
    im_size = tuple(get_global()["image_size"]) + (3,)

    data_aug = get_data_augmentation()

    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=im_size
    )

    base_model.trainable = True
    unfreeze = cfg["unfreeze_layers"]
    if unfreeze == 0:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-unfreeze]:
            layer.trainable = False

    model = keras.Sequential([
        keras.layers.Input(shape=im_size),
        data_aug,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(cfg["dense_units"], use_bias=False, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(cfg["dropout_rate"]),
        layers.Dense(get_global()["num_classes"], activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
