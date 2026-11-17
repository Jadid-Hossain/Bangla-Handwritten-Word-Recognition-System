"""Train the Bangla handwritten character model and record the run in MLflow.

The model classifies the 84 BanglaLekha-Isolated classes plus 10 synthetic
vowel-sign (kar) classes, so the Streamlit app can read whole words such as
"মানুষ". Run it twice with different settings to get the two MLflow runs:

    python train.py --run-name baseline --arch base --epochs 5
    python train.py --run-name deeper --arch deeper --epochs 8 --lr 0.0015

The dataset (BanglaLekha-Isolated, Mendeley doi:10.17632/hf6sf8zrkc.2) is
expected at data/BanglaLekha-Isolated/Images (folders 1..84). If only the zip
is present it is extracted first; if neither is present it is downloaded.
"""
import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from bangla_common import (
    BASE_DIR, IMG_SIZE, KAR_CLASSES, LABELS_PATH, MLFLOW_ARTIFACTS, MLFLOW_DIR,
    MODEL_PATH, NUM_CLASSES, TRAIN_EXPERIMENT, TRACKING_URI, load_labels,
    setup_device,
)
from data_prep import (
    KARS_PER_CLASS, N_REAL_CLASSES, WORKERS, build_cache, ensure_dataset,
    load_augmented_real,
)

DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "BanglaLekha-Isolated.zip"
DATASET_URL = ("https://data.mendeley.com/public-files/datasets/hf6sf8zrkc/files/"
               "ffe73a3a-5999-4ad2-970f-3e14c141cdb1/file_downloaded")
CACHE_PATH = DATA_DIR / f"cache_{IMG_SIZE}.npz"


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def conv_block(layers: list, filters: int, bn: bool) -> None:
    layers.append(tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
    if bn:
        layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"))
    if bn:
        layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.MaxPooling2D())


def build_model(arch: str) -> tf.keras.Sequential:
    layers: list = [tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
                    tf.keras.layers.Rescaling(1 / 255.0)]
    if arch == "base":
        conv_block(layers, 32, bn=False)
        conv_block(layers, 64, bn=False)
        layers += [tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
                   tf.keras.layers.MaxPooling2D(),
                   tf.keras.layers.Flatten(),
                   tf.keras.layers.Dense(192, activation="relu"),
                   tf.keras.layers.Dropout(0.35)]
    elif arch == "deeper":
        conv_block(layers, 32, bn=True)
        conv_block(layers, 64, bn=True)
        conv_block(layers, 128, bn=True)
        layers += [tf.keras.layers.GlobalAveragePooling2D(),
                   tf.keras.layers.Dense(256, activation="relu"),
                   tf.keras.layers.Dropout(0.35)]
    else:
        raise ValueError(f"unknown arch: {arch}")
    layers.append(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    return tf.keras.Sequential(layers)


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Send the per-epoch train/test numbers to the open MLflow run and keep the
    best model: whenever the test accuracy improves, the model file is updated."""

    def __init__(self, model_path: Path):
        super().__init__()
        self.model_path = model_path
        self.best = -1.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = float(logs.get("val_accuracy", 0.0))
        mlflow.log_metrics({
            "train_loss": logs.get("loss", 0.0),
            "train_accuracy": logs.get("accuracy", 0.0),
            "test_loss": logs.get("val_loss", 0.0),
            "test_accuracy": acc,
        }, step=epoch + 1)
        if acc > self.best:
            self.best = acc
            self.model.save(self.model_path)
            mlflow.log_metric("best_test_accuracy", acc, step=epoch + 1)
            print(f"  new best test accuracy {acc:.4f} -> saved {self.model_path.name}")


def confusion_figure(y_true: np.ndarray, y_pred: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, label="images")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix, test set (84 dataset classes + 10 vowel-sign classes)")
    fig.tight_layout()
    return fig


def stratified_split(y: np.ndarray, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Same split for every run, so the two MLflow runs are comparable."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])
    return np.concatenate(train_idx), np.concatenate(test_idx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Bangla word-recognition model")
    ap.add_argument("--run-name", default="baseline")
    ap.add_argument("--arch", default="base", choices=["base", "deeper"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--augment", type=int, default=0,
                    help="augmented copies of every real image (0 = off)")
    ap.add_argument("--prepare-only", action="store_true", help="build the data cache and exit")
    args = ap.parse_args()

    device = setup_device()
    labels = load_labels()  # validates labels.json against NUM_CLASSES

    if CACHE_PATH.exists():
        print(f"==> Loading cached data: {CACHE_PATH.name}")
    else:
        ensure_dataset(ZIP_PATH, DATA_DIR, DATASET_URL)
        build_cache(CACHE_PATH)
    with np.load(CACHE_PATH) as f:
        x, y = f["images"], f["labels"]
    if args.prepare_only:
        print("cache ready, --prepare-only set: nothing trained")
        return

    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    train_idx, test_idx = stratified_split(y, args.test_fraction, args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    real_train = y_train < N_REAL_CLASSES
    if args.augment > 0:
        print(f"==> Augmenting {int(real_train.sum())} training images x{args.augment}")
        for _ in range(args.augment):
            xa = load_augmented_real(x_train[real_train])
            x_train = np.concatenate([x_train, xa.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
            y_train = np.concatenate([y_train, y_train[real_train]])
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(x_train))
        x_train, y_train = x_train[order], y_train[order]
    print(f"train: {len(x_train)}  test: {len(x_test)}  classes: {NUM_CLASSES}")

    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(args.arch)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary(print_fn=lambda s: print(s))
    n_params = model.count_params()
    MODEL_PATH.parent.mkdir(exist_ok=True)  # the best-epoch callback saves here

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(TRAIN_EXPERIMENT)
    exp_id = exp.experiment_id if exp else mlflow.create_experiment(
        TRAIN_EXPERIMENT, artifact_location=(MLFLOW_ARTIFACTS / TRAIN_EXPERIMENT).as_uri())

    mlflow.end_run()  # close any run left open by an earlier attempt
    run = mlflow.start_run(experiment_id=exp_id, run_name=args.run_name)
    mlflow.log_params({
        "arch": args.arch, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "img_size": IMG_SIZE,
        "seed": args.seed, "device": device, "parameters": n_params,
        "classes": NUM_CLASSES, "dataset_classes": N_REAL_CLASSES,
        "synthetic_kar_classes": len(KAR_CLASSES),
        "kars_per_class": KARS_PER_CLASS, "workers": WORKERS,
        "augmented_copies": args.augment,
        "train_images": len(x_train), "test_images": len(x_test),
    })
    print(f"MLflow run started: {run.info.run_id}")

    start = time.time()
    with tf.device("/GPU:0" if device == "GPU" else "/CPU:0"):
        history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=[MLflowEpochLogger(MODEL_PATH)], verbose=2)
    seconds = time.time() - start
    print(f"\ntraining took {seconds / 60:.1f} minutes on the {device}")

    # the shipped model is the best-epoch checkpoint saved during training
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(x_test, verbose=0)
    preds = probs.argmax(axis=1)
    top3 = float((y_test[:, None] == np.argsort(-probs, axis=1)[:, :3]).any(axis=1).mean())
    accuracy = float((preds == y_test).mean())
    test_loss = float(model.evaluate(x_test, y_test, verbose=0)[0])
    print(f"saved model test accuracy: {accuracy:.4f}  (top-3: {top3:.4f})")

    mlflow.log_metrics({
        "final_test_accuracy": accuracy, "final_test_loss": test_loss,
        "final_top3_accuracy": top3, "train_seconds": seconds,
    })
    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact(str(LABELS_PATH))
    tmp_fig = DATA_DIR / "confusion_matrix_last_run.png"
    confusion_figure(y_test, preds).savefig(tmp_fig, dpi=110)
    mlflow.log_artifact(str(tmp_fig))
    mlflow.end_run()
    print(f"saved {MODEL_PATH.relative_to(BASE_DIR)} and finished MLflow run '{args.run_name}'")


if __name__ == "__main__":
    main()
