"""MobileNetV2 transfer-learning model, shared by train.py (any backend) and export_tflite.py (TensorFlow backend).

The model takes raw RGB pixels as float32 in [0, 255]; scaling to MobileNetV2's [-1, 1] happens inside the
graph, so the Pi only has to resize and pad the photo.
"""
import keras

import config as C


def build_model(img_size, weights="imagenet"):
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="to_minus1_1")(inp)
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                          weights=weights, pooling="avg")
    x = base(x, training=False)   # BatchNorm stays in inference mode even when layers are unfrozen
    x = keras.layers.Dropout(C.DROPOUT, name="head_dropout")(x)
    out = keras.layers.Dense(len(C.CLASSES), activation="softmax", name="probs")(x)
    return keras.Model(inp, out, name="battery_shape_mnv2"), base


def set_finetune(base):
    """Unfreeze from FT_UNFREEZE_FROM to the top; BatchNorm layers stay frozen."""
    base.trainable = True
    unfrozen = False
    for layer in base.layers:
        if layer.name == C.FT_UNFREEZE_FROM:
            unfrozen = True
        layer.trainable = unfrozen and not isinstance(layer, keras.layers.BatchNormalization)
    return sum(1 for l in base.layers if l.trainable)
