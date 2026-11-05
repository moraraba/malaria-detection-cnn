"""Grad-CAM: highlight which part of a cell image drove the model's prediction.

    python -m src.gradcam --model-path outputs/mobilenet.keras --n 8
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data import CLASS_NAMES, load_splits
from src.models import CAM_LAYER


def cam_layer_name(model):
    names = {l.name for l in model.layers}
    for layer in CAM_LAYER.values():
        if layer in names:
            return layer
    raise ValueError(f"No Grad-CAM layer found; expected one of {list(CAM_LAYER.values())}")


def make_gradcam_heatmap(model, image, layer_name=None):
    """Return (heatmap 0-1 at conv resolution, P(parasitized)) for one image (H, W, 3), 0-255.

    The heatmap explains the predicted class: evidence for "parasitized" when
    p >= 0.5, evidence for "uninfected" otherwise.
    """
    layer_name = layer_name or cam_layer_name(model)
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    batch = tf.convert_to_tensor(image[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(batch, training=False)
        p = pred[:, 0]
        score = tf.where(p >= 0.5, p, 1.0 - p)
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # one weight per feature map
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights[0], axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy(), float(p[0])


def overlay(image, heatmap, alpha=0.4):
    """Blend a heatmap onto an image (both returned as uint8 RGB)."""
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape[:2]
    hm = tf.image.resize(heatmap[..., None], (h, w)).numpy()[..., 0]
    colored = plt.get_cmap("jet")(hm)[..., :3] * 255
    return np.clip(image * (1 - alpha) + colored * alpha, 0, 255).astype(np.uint8)


def save_grid(model, ds, out_path, n=8):
    images, labels = [], []
    for x, y in ds.unbatch().take(n):
        images.append(x.numpy())
        labels.append(int(y.numpy()))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    for i, (img, lab) in enumerate(zip(images, labels)):
        hm, p = make_gradcam_heatmap(model, img)
        pred = int(p >= 0.5)
        axes[0, i].imshow(img.astype(np.uint8))
        axes[0, i].set_title(f"True: {CLASS_NAMES[lab]}", fontsize=8)
        axes[1, i].imshow(overlay(img, hm))
        colour = "green" if pred == lab else "red"
        axes[1, i].set_title(f"Pred: {CLASS_NAMES[pred]} ({p:.2f})", fontsize=8, color=colour)
        for ax in axes[:, i]:
            ax.axis("off")
    fig.suptitle("Grad-CAM: where the model looked (bottom row)", fontsize=10)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--data", default="tfds")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")
    a = p.parse_args()

    _, _, test, _ = load_splits(a.data, a.img_size, 32, a.seed)
    model = keras.models.load_model(a.model_path)
    out = Path(a.out_dir) / "figures" / f"{Path(a.model_path).stem}_gradcam.png"
    save_grid(model, test, out, a.n)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
