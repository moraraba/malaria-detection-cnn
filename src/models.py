"""Two models: a small CNN trained from scratch, and MobileNetV2 transfer learning."""
from tensorflow import keras
from tensorflow.keras import layers

# Layer whose output Grad-CAM explains, for each model
CAM_LAYER = {"baseline": "last_conv", "mobilenet": "out_relu"}
# MobileNetV2 layers from this one onwards are unfrozen when fine-tuning
FINE_TUNE_FROM = "block_13_expand"


def augmentation():
    """Random flips, rotations and zooms. Cells look the same at any angle,
    and the black fill matches the background of the smear images."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.25, fill_mode="constant"),
            layers.RandomZoom(0.1, fill_mode="constant"),
        ],
        name="augmentation",
    )


def build_baseline_cnn(img_size=128):
    """Three convolution blocks, then a single sigmoid output = P(parasitized)."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for i, filters in enumerate([32, 64, 128]):
        x = layers.Conv2D(
            filters, 3, padding="same", activation="relu",
            name="last_conv" if i == 2 else f"conv_{i + 1}",
        )(x)
        # momentum 0.9 (default 0.99) lets the running statistics settle within a few epochs
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_parasitized")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(img_size=128, weights="imagenet"):
    """MobileNetV2 pretrained on ImageNet, frozen, with a new classification head.

    The base is built on our own input tensor so its layers sit directly in
    this model; that keeps Grad-CAM simple (no nested model to unpack).
    """
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation()(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)  # MobileNetV2 expects [-1, 1]
    base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_tensor=x)
    if weights is not None:  # freeze pretrained features; random weights must be trained
        for layer in base.layers:
            layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_parasitized")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def unfreeze_top(model, from_layer=FINE_TUNE_FROM):
    """Unfreeze the last MobileNetV2 blocks for fine-tuning. BatchNorm layers stay
    frozen, as recommended, so their statistics are not disturbed."""
    names = [l.name for l in model.layers]
    start = names.index(from_layer)
    for i, layer in enumerate(model.layers):
        trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
