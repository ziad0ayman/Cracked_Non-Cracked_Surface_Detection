from tensorflow import keras
from tensorflow.keras import layers, regularizers
from src.utils.data_loader import get_data_augmentation

def build_rnn_column(input_shape=(224, 224, 3),
                     lstm_units=128, dense_units=384,
                     dropout_rate=0.485, lr=0.000133):
    data_aug = get_data_augmentation()

    inputs = keras.layers.Input(shape=input_shape)
    x = data_aug(inputs)
    x = layers.AveragePooling2D((4, 4))(x)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((56, 56 * 3))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(2, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
