import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

def load_datasets(data_root, image_size=(224, 224), batch_size=64, seed=42):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, label_mode='categorical', batch_size=batch_size,
        image_size=image_size, seed=seed)
    val = tf.keras.utils.image_dataset_from_directory(
        val_dir, label_mode='categorical', batch_size=batch_size,
        image_size=image_size, shuffle=False)
    test = tf.keras.utils.image_dataset_from_directory(
        test_dir, label_mode='categorical', batch_size=batch_size,
        image_size=image_size, shuffle=False)

    return train, val, test

def prefetch_datasets(train, val, test):
    train = train.prefetch(buffer_size=tf.data.AUTOTUNE)
    val = val.prefetch(buffer_size=tf.data.AUTOTUNE)
    test = test.prefetch(buffer_size=tf.data.AUTOTUNE)
    return train, val, test

def compute_class_weights(train_dir):
    class_names = sorted(os.listdir(train_dir))
    y_train = []
    for i, name in enumerate(class_names):
        class_path = os.path.join(train_dir, name)
        n = len(os.listdir(class_path))
        y_train.extend([i] * n)
    y_train = np.array(y_train)
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

def get_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1)
    ], name='data_augmentation')
