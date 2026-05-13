from tensorflow import keras
from tensorflow.keras import layers
from src.utils.data_loader import get_data_augmentation
from src.utils.config import get_model, get_global

def build_ffnn(**kwargs):
    cfg = {**get_model("ffnn"), **kwargs}
    im_size = tuple(get_global()["image_size"]) + (3,)

    data_aug = get_data_augmentation()

    model = keras.Sequential([
        layers.Input(shape=im_size),
        data_aug,
        layers.Flatten(),
        layers.Dense(cfg["dense_units"], activation='relu'),
        layers.Dropout(cfg["dropout_rate"]),
        layers.Dense(get_global()["num_classes"], activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model
