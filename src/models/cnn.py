from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation

def build_cnn(input_shape=(224, 224, 3),
              n_blocks=6, base_filters=64,
              dense_units=128, dropout_rate=0.254,
              lr=0.000517):
    data_aug = get_data_augmentation()

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(data_aug)

    for i in range(n_blocks):
        filters = min(base_filters * (2 ** i), 512)
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, use_bias=False, kernel_regularizer=regularizers.l2(1e-5)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
