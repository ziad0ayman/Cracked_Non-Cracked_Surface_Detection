from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation
from src.utils.config import get_model, get_global

def build_cnn(**kwargs):
    cfg = {**get_model("cnn"), **kwargs}
    im_size = tuple(get_global()["image_size"]) + (3,)

    data_aug = get_data_augmentation()

    model = keras.Sequential()
    model.add(layers.Input(shape=im_size))
    model.add(data_aug)

    for i in range(cfg["n_blocks"]):
        filters = min(cfg["base_filters"] * (2 ** i), 512)
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(cfg["dense_units"], use_bias=False, kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(cfg["dropout_rate"]))
    model.add(layers.Dense(get_global()["num_classes"], activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
