import argparse
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

from src.utils.config import get_model, get_global
from src.utils.data_loader import load_datasets, prefetch_datasets, compute_class_weights
from src.utils.evaluate import plot_training_history, plot_confusion_matrix
from src.models.transfer import build_transfer


def main(args):
    model_cfg = get_model("transfer")
    global_cfg = get_global()
    im_size = tuple(global_cfg["image_size"])

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

    batch_size = args.batch_size or model_cfg["batch_size"]
    epochs = args.epochs or model_cfg["epochs"]
    train, val, test = load_datasets(data_root, image_size=im_size, batch_size=batch_size)
    train, val, test = prefetch_datasets(train, val, test)

    class_weights = compute_class_weights(os.path.join(data_root, 'train'))

    model = build_transfer()
    model.summary()

    ckpt_path = os.path.join(args.checkpoint_dir, 'transfer_best.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                            monitor=global_cfg["monitor"], mode='min'),
        tf.keras.callbacks.EarlyStopping(patience=global_cfg["early_stopping_patience"],
                                          restore_best_weights=True,
                                          monitor=global_cfg["monitor"], mode='min'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=global_cfg["reduce_lr_factor"],
                                              patience=global_cfg["reduce_lr_patience"],
                                              monitor=global_cfg["monitor"], mode='min'),
        tf.keras.callbacks.CSVLogger(os.path.join(args.results_dir, 'transfer_training.csv'))
    ]

    initial_epoch = 0
    if args.resume and os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        initial_epoch = args.resume_epoch
        print(f'Resuming from epoch {initial_epoch}')

    history = model.fit(
        train, validation_data=val, epochs=epochs,
        class_weight=class_weights, callbacks=callbacks,
        initial_epoch=initial_epoch
    )

    test_loss, test_acc = model.evaluate(test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}')

    model.save(os.path.join(args.checkpoint_dir, 'transfer_final.keras'))

    plot_training_history(history, save_path=os.path.join(args.results_dir, 'transfer_history.png'))

    y_true = np.concatenate([y.numpy().argmax(axis=1) for _, y in test])
    y_pred = np.argmax(model.predict(test), axis=1)
    class_names = sorted(os.listdir(os.path.join(data_root, 'train')))
    plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path=os.path.join(args.results_dir, 'transfer_cm.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/split')
    parser.add_argument('--results-dir', default='results/transfer')
    parser.add_argument('--checkpoint-dir', default='models')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-epoch', type=int, default=0)
    main(parser.parse_args())
