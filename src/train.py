"""Train a model, then evaluate it on the test set and save Grad-CAM examples.

    python -m src.train --model baseline
    python -m src.train --model mobilenet --epochs 10 --fine-tune-epochs 5
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from src.data import load_splits
from src.evaluate import evaluate
from src.gradcam import save_grid
from src.models import build_baseline_cnn, build_transfer_model, compile_model, unfreeze_top


def plot_history(history, out_path, fine_tune_start=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ["loss", "accuracy", "recall"]):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="validation")
        if fine_tune_start:
            ax.axvline(fine_tune_start - 0.5, color="grey", linestyle="--", linewidth=1)
            ax.text(fine_tune_start - 0.4, ax.get_ylim()[1], " fine-tune", va="top", fontsize=8, color="grey")
        ax.set_title(key.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["baseline", "mobilenet"], default="baseline")
    p.add_argument("--data", default="tfds", help='"tfds" or a folder with Parasitized/ and Uninfected/')
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--fine-tune-epochs", type=int, default=5, help="MobileNet only")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-pretrained", action="store_true", help="MobileNet with random weights, for offline code testing only (no fine-tune step)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args()

    keras.utils.set_random_seed(a.seed)
    out = Path(a.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    train, val, test, sizes = load_splits(a.data, a.img_size, a.batch_size, a.seed)
    print(f"Images - train: {sizes['train']:,}  validation: {sizes['val']:,}  test: {sizes['test']:,}")

    if a.model == "baseline":
        model = build_baseline_cnn(a.img_size)
    else:
        model = build_transfer_model(a.img_size, weights=None if a.no_pretrained else "imagenet")
    compile_model(model, a.lr)

    ckpt = out / f"{a.model}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", mode="min", save_best_only=True),
    ]
    hist = model.fit(train, validation_data=val, epochs=a.epochs, callbacks=callbacks).history

    fine_tune_start = None
    # Fine-tuning only makes sense on top of pretrained weights
    if a.model == "mobilenet" and a.fine_tune_epochs > 0 and not a.no_pretrained:
        fine_tune_start = len(hist["loss"])
        unfreeze_top(model)
        compile_model(model, a.lr / 100)  # much smaller steps so pretrained weights are not wrecked
        callbacks[1] = keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_loss", mode="min", save_best_only=True,
            initial_value_threshold=min(hist["val_loss"]),
        )
        ft = model.fit(train, validation_data=val, initial_epoch=fine_tune_start,
                       epochs=fine_tune_start + a.fine_tune_epochs, callbacks=callbacks).history
        for k in hist:
            hist[k] = hist[k] + ft.get(k, [])

    # The checkpoint holds the best model across both phases (lowest validation loss)
    model = keras.models.load_model(ckpt)
    (out / f"{a.model}_history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, out / "figures" / f"{a.model}_training.png", fine_tune_start)

    metrics = evaluate(model, test, out, a.model)
    save_grid(model, test, out / "figures" / f"{a.model}_gradcam.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
