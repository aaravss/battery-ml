"""Step 6: export a trained variant to TFLite (float32 + full-integer int8).  python export_tflite.py --variant X

Runs on the TensorFlow backend (the converter needs TF), rebuilding the same Keras graph and loading the
backend-agnostic .weights.h5 written by train.py. The int8 model takes uint8 pixels (0-255) directly and returns
float32 probabilities, so the Pi code is identical for both files apart from the input dtype.
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import keras
from ai_edge_litert.interpreter import Interpreter

import config as C
from imgutil import load_image
from model_def import build_model


def run_tflite(path, X):
    it = Interpreter(model_path=str(path), num_threads=8)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in X:
        it.set_tensor(i["index"], x[None].astype(i["dtype"]))
        it.invoke()
        y = it.get_tensor(o["index"])[0].astype(np.float32)
        if o["dtype"] != np.float32:
            s, z = o["quantization"]
            y = (y - z) * s
        out.append(y)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    args = ap.parse_args()
    run = C.RUNS / args.variant
    done = json.load(open(run / "done.json"))
    size = done["img_size"]
    out = run / "export"
    if (out / "export.json").exists() and (out / "export.json").stat().st_mtime > (run / "best.weights.h5").stat().st_mtime:
        print(f"[export:{args.variant}] up to date; skipping")
        return
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    model, _ = build_model(size, weights=None)
    model.load_weights(run / "best.weights.h5")

    probe_paths = json.load(open(run / "probe_paths.json"))
    Xp = np.stack([load_image(C.ROOT / p, size) for p in probe_paths]).astype(np.float32)
    ref = np.load(run / "probe_probs.npy")
    tf_probs = model.predict(Xp, verbose=0)
    print(f"  cross-backend check (torch-trained vs TF-rebuilt): max |dp| = {np.abs(tf_probs - ref).max():.2e}")

    model.export(str(out / "saved_model"), format="tf_saved_model", verbose=False)

    conv = tf.lite.TFLiteConverter.from_saved_model(str(out / "saved_model"))
    (out / "battery_float32.tflite").write_bytes(conv.convert())

    rows = [r for r in csv.DictReader(open(C.WORK / "splits.csv", encoding="utf-8"))
            if r["split"] == "train" and r["used"] == "1"]
    rng = np.random.default_rng(C.SEED)
    rep = [rows[i] for i in rng.permutation(len(rows))[:300]]   # real, un-augmented train images only

    def representative():
        for r in rep:
            yield [load_image(C.ROOT / r["path"], size)[None].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(out / "saved_model"))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.float32
    (out / "battery_int8.tflite").write_bytes(conv.convert())

    info = {"img_size": size, "classes": C.CLASSES,
            "cross_backend_max_abs_diff": float(np.abs(tf_probs - ref).max())}
    for name in ("battery_float32", "battery_int8"):
        p = run_tflite(out / f"{name}.tflite", Xp)
        info[f"{name}_vs_keras_max_abs_diff"] = float(np.abs(p - tf_probs).max())
        info[f"{name}_size_kb"] = round((out / f"{name}.tflite").stat().st_size / 1024, 1)
        print(f"  {name}: {info[f'{name}_size_kb']} KB, max |dp| vs Keras = {info[f'{name}_vs_keras_max_abs_diff']:.3f}")
    shutil.rmtree(out / "saved_model")
    json.dump(info, open(out / "export.json", "w"), indent=2)
    print(f"[export:{args.variant}] ok")


if __name__ == "__main__":
    main()
