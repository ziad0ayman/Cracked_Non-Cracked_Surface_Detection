from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation

def build_rnn_md(input_shape=(224, 224, 3),
                 lstm_units=128, dense_units=512,
                 dropout_rate=0.262, lr=6.62e-5):
    data_aug = get_data_augmentation()

    inputs = keras.layers.Input(shape=input_shape)
    x = data_aug(inputs)
    x = layers.AveragePooling2D((4, 4))(x)

    x_rows = layers.Reshape((56, 56 * 3))(x)
    lstm_rows = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x_rows)

    x_cols = layers.Permute((2, 1, 3))(x)
    x_cols = layers.Reshape((56, 56 * 3))(x_cols)
    lstm_cols = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x_cols)

    merged = layers.Concatenate()([lstm_rows, lstm_cols])
    x_head = layers.BatchNormalization()(merged)
    x_head = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x_head)
    x_head = layers.Dropout(dropout_rate)(x_head)
    outputs = layers.Dense(2, activation='softmax')(x_head)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
