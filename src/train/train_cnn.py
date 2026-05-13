import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import tensorflow as tf

from src.utils.data_loader import load_datasets, prefetch_datasets, compute_class_weights
from src.utils.evaluate import plot_training_history, plot_confusion_matrix
from src.models.cnn import build_cnn


def main(args):
    data_root = args.data_root
    train_dir = os.path.join(data_root, 'train')
    from src.utils.data_setup import main as setup_data
    setup_data()
    if not os.path.isdir(train_dir):
        print(f"ERROR: Data still missing at '{train_dir}' after automatic setup.")
        print("Check your internet connection (Kaggle download) and config.")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train, val, test = load_datasets(data_root, image_size=(224, 224), batch_size=64)
    train, val, test = prefetch_datasets(train, val, test)

    class_weights = compute_class_weights(os.path.join(data_root, 'train'))

    model = build_cnn()
    model.summary()

    ckpt_path = os.path.join(args.checkpoint_dir, 'cnn_best.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                            monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True,
                                          monitor='val_loss', mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5,
                                              monitor='val_loss', mode='min'),
        tf.keras.callbacks.CSVLogger(os.path.join(args.results_dir, 'cnn_training.csv'))
    ]

    initial_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        initial_epoch = args.resume_epoch
        print(f'Resuming from epoch {initial_epoch}')

    history = model.fit(
        train, validation_data=val, epochs=args.epochs,
        class_weight=class_weights, callbacks=callbacks,
        initial_epoch=initial_epoch
    )

    test_loss, test_acc = model.evaluate(test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')

    model.save(os.path.join(args.checkpoint_dir, 'cnn_final.keras'))

    plot_training_history(history, save_path=os.path.join(args.results_dir, 'cnn_history.png'))

    y_true = np.concatenate([y.numpy().argmax(axis=1) for _, y in test])
    y_pred = np.argmax(model.predict(test), axis=1)
    class_names = sorted(os.listdir(os.path.join(data_root, 'train')))
    plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path=os.path.join(args.results_dir, 'cnn_cm.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/split')
    parser.add_argument('--results-dir', default='results/cnn')
    parser.add_argument('--checkpoint-dir', default='models')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-epoch', type=int, default=0)
    main(parser.parse_args())
