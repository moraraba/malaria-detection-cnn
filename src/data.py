"""Load the NIH malaria cell images as train / validation / test tf.data pipelines.

Two sources are supported:
  * "tfds"  - downloads the dataset through TensorFlow Datasets (easiest on Colab)
  * a folder path - the Kaggle / NIH zip unpacked, with one sub-folder per class:
        cell_images/Parasitized/*.png
        cell_images/Uninfected/*.png

Label 1 = parasitized (infected), label 0 = uninfected. Parasitized is the
"positive" class, so recall = the share of infected cells the model catches.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["uninfected", "parasitized"]
SPLITS = (0.70, 0.15, 0.15)  # train, validation, test
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _load_tfds(data_dir=None):
    import tensorflow_datasets as tfds

    train_end = int(SPLITS[0] * 100)
    val_end = int((SPLITS[0] + SPLITS[1]) * 100)
    splits = [f"train[:{train_end}%]", f"train[{train_end}%:{val_end}%]", f"train[{val_end}%:]"]
    datasets, info = tfds.load(
        "malaria", split=splits, as_supervised=True, with_info=True, data_dir=data_dir
    )
    parasitized = info.features["label"].names.index("parasitized")

    def relabel(image, label):
        return image, tf.cast(tf.equal(label, parasitized), tf.int32)

    sizes = [info.splits[s].num_examples for s in splits]
    return [ds.map(relabel, num_parallel_calls=AUTOTUNE) for ds in datasets], sizes


def _list_folder(folder):
    paths, labels = [], []
    for class_dir in sorted(Path(folder).iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.lower()
        if name not in CLASS_NAMES:
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(p))
                labels.append(CLASS_NAMES.index(name))
    if not paths:
        raise FileNotFoundError(
            f"No images found in {folder}. Expected sub-folders named Parasitized and Uninfected."
        )
    return np.array(paths), np.array(labels, dtype=np.int32)


def _load_folder(folder, seed):
    paths, labels = _list_folder(folder)
    order = np.random.default_rng(seed).permutation(len(paths))
    paths, labels = paths[order], labels[order]
    n_train = int(SPLITS[0] * len(paths))
    n_val = int(SPLITS[1] * len(paths))
    cuts = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, len(paths))]

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return image, label

    datasets = [
        tf.data.Dataset.from_tensor_slices((paths[a:b], labels[a:b])).map(
            decode, num_parallel_calls=AUTOTUNE
        )
        for a, b in cuts
    ]
    return datasets, [b - a for a, b in cuts]


def _prepare(ds, img_size, batch_size, shuffle, seed):
    def resize(image, label):
        image = tf.image.resize(tf.cast(image, tf.float32), (img_size, img_size))
        return image, label  # pixel values stay 0-255; each model rescales itself

    ds = ds.map(resize, num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def load_splits(source="tfds", img_size=128, batch_size=64, seed=42, data_dir=None):
    """Return (train, val, test, sizes). Images are float32 0-255, labels 0/1.

    The split is fixed by `seed` (folder) or by TFDS slicing, so train.py and
    evaluate.py always see the same test images.
    """
    if source == "tfds":
        raw, sizes = _load_tfds(data_dir)
    else:
        raw, sizes = _load_folder(source, seed)
    train, val, test = raw
    return (
        _prepare(train, img_size, batch_size, shuffle=True, seed=seed),
        _prepare(val, img_size, batch_size, shuffle=False, seed=seed),
        _prepare(test, img_size, batch_size, shuffle=False, seed=seed),
        dict(zip(["train", "val", "test"], sizes)),
    )
