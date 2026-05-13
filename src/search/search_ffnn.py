import os
import json
import optuna
from optuna.integration import TFKerasPruningCallback

from src.utils.config import get_search_space, get_global
from src.utils.data_loader import load_datasets, prefetch_datasets, compute_class_weights
from src.models.ffnn import build_ffnn


def objective(trial):
    space = get_search_space("ffnn")
    im_size = tuple(get_global()["image_size"])

    dense_units = trial.suggest_int("dense_units", space["dense_units"]["min"],
                                    space["dense_units"]["max"], step=space["dense_units"]["step"])
    dropout_rate = trial.suggest_float("dropout_rate", space["dropout_rate"]["min"],
                                       space["dropout_rate"]["max"])
    lr = trial.suggest_float("learning_rate", space["learning_rate"]["min"],
                             space["learning_rate"]["max"], log=space["learning_rate"]["log"])

    model = build_ffnn(dense_units=dense_units, dropout_rate=dropout_rate, learning_rate=lr)
    model.summary()

    train_dir = os.path.join("data/split", "train")
    train, val, _ = load_datasets("data/split", image_size=im_size, batch_size=32)
    train, val, _ = prefetch_datasets(train, val, None)
    class_weights = compute_class_weights(train_dir)

    callbacks = [TFKerasPruningCallback(trial, "val_accuracy")]
    model.fit(train, validation_data=val, epochs=20, callbacks=callbacks,
              class_weight=class_weights, verbose=0)
    val_loss, val_acc = model.evaluate(val, verbose=0)
    return val_acc


if __name__ == "__main__":
    from src.utils.data_setup import main as setup_data
    setup_data()

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner())
    n_trials = get_global().get("n_trials", 30)
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    os.makedirs("results", exist_ok=True)
    with open("results/ffnn_optuna_best.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
