from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation
from src.utils.config import get_model, get_global

def build_rnn_md(**kwargs):
    cfg = {**get_model("rnn_md"), **kwargs}
    im_size = tuple(get_global()["image_size"]) + (3,)

    data_aug = get_data_augmentation()

    inputs = keras.layers.Input(shape=im_size)
    x = data_aug(inputs)
    x = layers.AveragePooling2D((4, 4))(x)

    x_rows = layers.Reshape((56, 56 * 3))(x)
    lstm_rows = layers.Bidirectional(layers.LSTM(cfg["lstm_units"], return_sequences=False))(x_rows)

    x_cols = layers.Permute((2, 1, 3))(x)
    x_cols = layers.Reshape((56, 56 * 3))(x_cols)
    lstm_cols = layers.Bidirectional(layers.LSTM(cfg["lstm_units"], return_sequences=False))(x_cols)

    merged = layers.Concatenate()([lstm_rows, lstm_cols])
    x_head = layers.BatchNormalization()(merged)
    x_head = layers.Dense(cfg["dense_units"], activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x_head)
    x_head = layers.Dropout(cfg["dropout_rate"])(x_head)
    outputs = layers.Dense(get_global()["num_classes"], activation='softmax')(x_head)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
