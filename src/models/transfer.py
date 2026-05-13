import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation

def build_transfer(input_shape=(224, 224, 3),
                   dense_units=384, dropout_rate=0.423,
                   lr=0.000298, unfreeze_layers=20):
    data_aug = get_data_augmentation()

    base_model = tf.keras.applications.DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = True
    if unfreeze_layers == 0:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        data_aug,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, use_bias=False, kernel_regularizer=regularizers.l2(1e-5)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(dropout_rate),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
