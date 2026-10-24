"""Step 5: train one variant.  python train.py --variant baseline|synthetic

Phase 1 trains only the new head on a frozen ImageNet MobileNetV2; phase 2 fine-tunes the top blocks at a low LR.
Crash-safe: BackupAndRestore resumes mid-phase, phase markers skip finished phases, and the run directory is
wiped automatically when the data split or training config changes (so adding photos always retrains).
Backend: PyTorch+CUDA when a GPU is usable (TensorFlow has no native-Windows GPU support), else TensorFlow CPU.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def pick_backend():
    if os.environ.get("KERAS_BACKEND"):
        return os.environ["KERAS_BACKEND"], "env override"
    try:
        import torch
        if torch.cuda.is_available():
            return "torch", torch.cuda.get_device_name(0)
    except Exception as e:  # torch missing or broken CUDA: fall back to CPU
        print("  torch/CUDA unavailable:", e)
    return "tensorflow", "CPU"


BACKEND, DEVICE = pick_backend()
os.environ["KERAS_BACKEND"] = BACKEND
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")   # tf.data only; never let TF grab the GPU
import keras

from imgutil import load_image
from model_def import build_model, set_finetune

IMG = C.IMG_SIZE if BACKEND == "torch" else C.IMG_SIZE_CPU


def read_rows(variant):
    rows = [r for r in csv.DictReader(open(C.WORK / "splits.csv", encoding="utf-8")) if r["used"] == "1"]
    if variant == "synthetic":
        syn = C.WORK / "synthetic.csv"
        rows += [r for r in csv.DictReader(open(syn, encoding="utf-8"))] if syn.exists() else []
    return rows


def load_all(rows):
    with ThreadPoolExecutor(12) as ex:
        return np.stack(list(ex.map(lambda r: load_image(C.ROOT / r["path"], IMG), rows)))


def make_augment(size):
    c = (size - 1) / 2.0

    def blur(x):
        sigma = tf.random.uniform([], 0.4, 1.8)
        r = tf.range(-3.0, 4.0)
        k = tf.exp(-r ** 2 / (2 * sigma ** 2))
        k = k / tf.reduce_sum(k)
        kx = tf.tile(tf.reshape(k, [1, 7, 1, 1]), [1, 1, 3, 1])
        ky = tf.tile(tf.reshape(k, [7, 1, 1, 1]), [1, 1, 3, 1])
        x = tf.pad(x[None], [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        x = tf.nn.depthwise_conv2d(x, kx, [1, 1, 1, 1], "VALID")
        x = tf.nn.depthwise_conv2d(x, ky, [1, 1, 1, 1], "VALID")
        return x[0]

    def aug(img, y, w):
        img = tf.cast(img, tf.float32)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        # rotation (any angle: batteries lie in the tray at any orientation) + zoom + shift, one affine warp
        ang = tf.random.uniform([], -np.pi, np.pi)
        zoom = tf.random.uniform([], 0.85, 1.15)
        tx = tf.random.uniform([], -0.1, 0.1) * size
        ty = tf.random.uniform([], -0.1, 0.1) * size
        cs, sn = tf.cos(ang) / zoom, tf.sin(ang) / zoom
        t = tf.stack([cs, -sn, c - cs * c + sn * c - tx, sn, cs, c - sn * c - cs * c - ty, 0.0, 0.0])
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[None], transforms=t[None], output_shape=[size, size], fill_value=0.0,
            interpolation="BILINEAR", fill_mode="REFLECT")[0]
        img = tf.image.random_brightness(img, 0.2 * 255)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.cond(tf.random.uniform([]) < 0.3, lambda: blur(img), lambda: img)
        img = tf.clip_by_value(img, 0.0, 255.0)
        img.set_shape([size, size, 3])
        return img, y, w

    return aug


def fingerprint(variant):
    h = hashlib.sha1()
    h.update((C.WORK / "splits.csv").read_bytes())
    if variant == "synthetic" and (C.WORK / "synthetic.csv").exists():
        h.update((C.WORK / "synthetic.csv").read_bytes())
    for k in ("IMG_SIZE", "BATCH", "HEAD_EPOCHS", "FT_EPOCHS", "HEAD_LR", "FT_LR", "FT_UNFREEZE_FROM", "DROPOUT",
              "LABEL_SMOOTHING", "SEED", "CLASS_WEIGHT_CLIP", "CLASSES"):
        h.update(repr(getattr(C, k)).encode())
    h.update(BACKEND.encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline", choices=["baseline", "synthetic"])
    ap.add_argument("--max-epochs", type=int, default=None, help="smoke-test override")
    args = ap.parse_args()
    run = C.RUNS / args.variant
    fp = fingerprint(args.variant)
    if (run / "fingerprint.txt").exists() and (run / "fingerprint.txt").read_text() != fp:
        print("  data/config changed since last run -> starting this variant from scratch")
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "fingerprint.txt").write_text(fp)
    if (run / "done.json").exists():
        print(f"[train:{args.variant}] already complete for this data; skipping")
        return

    keras.utils.set_random_seed(C.SEED)
    print(f"[train:{args.variant}] backend={BACKEND} device={DEVICE} img={IMG}")
    rows = read_rows(args.variant)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    cidx = {c: i for i, c in enumerate(C.CLASSES)}
    ytr = np.array([cidx[r["cls"]] for r in tr])
    yva = np.array([cidx[r["cls"]] for r in va])
    counts = np.bincount(ytr, minlength=len(C.CLASSES))
    cw = np.where(counts > 0, len(ytr) / (len(C.CLASSES) * np.maximum(counts, 1)), 0.0)
    cw = np.clip(cw, *C.CLASS_WEIGHT_CLIP)
    print("  train counts:", dict(zip(C.CLASSES, counts.tolist())), "val counts:",
          dict(zip(C.CLASSES, np.bincount(yva, minlength=len(C.CLASSES)).tolist())))
    print("  class weights:", dict(zip(C.CLASSES, np.round(cw, 2).tolist())))

    t0 = time.time()
    Xtr, Xva = load_all(tr), load_all(va)
    print(f"  loaded {len(Xtr)} train / {len(Xva)} val images in {time.time() - t0:.0f}s")
    onehot = np.eye(len(C.CLASSES), dtype=np.float32)
    wtr = cw[ytr].astype(np.float32)
    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr, onehot[ytr], wtr))
                .shuffle(len(Xtr), seed=C.SEED, reshuffle_each_iteration=True)
                .map(make_augment(IMG), num_parallel_calls=tf.data.AUTOTUNE)
                .batch(C.BATCH).prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((Xva.astype(np.float32), onehot[yva])).batch(C.BATCH)

    model, base = build_model(IMG)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=C.LABEL_SMOOTHING)
    metrics = [keras.metrics.CategoricalAccuracy(name="acc")]
    head_epochs = args.max_epochs or C.HEAD_EPOCHS
    ft_epochs = args.max_epochs or C.FT_EPOCHS

    # ---- phase 1: head only ----
    if not (run / "head.weights.h5").exists():
        base.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(C.HEAD_LR), loss=loss, metrics=metrics)
        model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, verbose=2, callbacks=[
            keras.callbacks.BackupAndRestore(str(run / "backup_head")),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.CSVLogger(str(run / "log_head.csv"), append=True),
        ])
        model.save_weights(run / "head.weights.h5")
    else:
        print("  phase 1 already done; loading head weights")
        model.load_weights(run / "head.weights.h5")

    # ---- phase 2: fine-tune top blocks ----
    n_trainable = set_finetune(base)
    print(f"  fine-tuning {n_trainable} base layers from '{C.FT_UNFREEZE_FROM}' at lr={C.FT_LR}")
    model.compile(optimizer=keras.optimizers.Adam(C.FT_LR), loss=loss, metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs, verbose=2, callbacks=[
        keras.callbacks.BackupAndRestore(str(run / "backup_ft")),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7),
        keras.callbacks.CSVLogger(str(run / "log_ft.csv"), append=True),
    ])
    model.save_weights(run / "best.weights.h5")

    probe = model.predict(Xva[:32].astype(np.float32), verbose=0)
    np.save(run / "probe_probs.npy", probe)   # export_tflite.py checks the TF-backend/TFLite model reproduces these
    json.dump([r["path"] for r in va[:32]], open(run / "probe_paths.json", "w"))
    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    done = dict(variant=args.variant, backend=BACKEND, device=DEVICE, img_size=IMG, classes=C.CLASSES,
                train_counts=dict(zip(C.CLASSES, counts.tolist())), class_weights=dict(zip(C.CLASSES, cw.tolist())),
                val=val_eval, minutes=round((time.time() - t0) / 60, 1))
    json.dump(done, open(run / "done.json", "w"), indent=2)
    print(f"[train:{args.variant}] done: val={val_eval} in {done['minutes']} min")


if __name__ == "__main__":
    main()
