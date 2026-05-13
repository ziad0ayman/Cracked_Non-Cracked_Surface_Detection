from tensorflow import keras
from tensorflow.keras import layers
from src.utils.data_loader import get_data_augmentation

def build_ffnn(input_shape=(224, 224, 3), dense_units=512, dropout_rate=0.157, lr=2e-5):
    data_aug = get_data_augmentation()

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        data_aug,
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model
