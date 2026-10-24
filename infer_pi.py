#!/usr/bin/env python3
"""Classify the battery SHAPE in one photo with the exported TFLite model (Raspberry Pi 4, offline).

    python3 infer_pi.py photo.jpg
    python3 infer_pi.py photo.jpg --model battery_float32.tflite --json

Prints the class and confidence. If confidence is below the calibrated threshold (model_info.json), the result
is reported as UNCERTAIN; the device should then treat the item as unidentified (hazardous by default).

Runtime: ai-edge-litert (Google's LiteRT, the successor of tflite-runtime). On Debian 13 / Raspberry Pi OS
"trixie" (64-bit, Python 3.13) install it inside a venv while you still have internet:
    python3 -m venv ~/battery-venv && ~/battery-venv/bin/pip install ai-edge-litert pillow
tflite-runtime is NOT an option there: its last release (2.14, 2023) has no wheels for Python > 3.11.
A 32-bit (armhf) OS has no ai-edge-litert wheels: use the 64-bit OS.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # older setups
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        sys.exit("Install the runtime first: pip install ai-edge-litert pillow")

HERE = Path(__file__).resolve().parent


# --- identical to pipeline/imgutil.py: training and inference must preprocess the same way ---
def pad_to_square(im):
    w, h = im.size
    if w == h:
        return im
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    s = max(w, h)
    out = Image.new("RGB", (s, s), fill)
    out.paste(im, ((s - w) // 2, (s - h) // 2))
    return out


def load_image(path, size):
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = pad_to_square(im.convert("RGB"))
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.uint8)
# ---------------------------------------------------------------------------------------------


def find(name, *dirs):
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("--model", help="default: model_info.json's recommended model next to this script or in models/")
    ap.add_argument("--labels", help="default: labels.txt next to the model")
    ap.add_argument("--threshold", type=float, help="override the calibrated 'uncertain' threshold")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", action="store_true", help="print one JSON line (for the rule-based layer)")
    a = ap.parse_args()

    info_path = find("model_info.json", HERE, HERE / "models")
    info = json.load(open(info_path)) if info_path else {}
    base = info_path.parent if info_path else HERE
    model = Path(a.model) if a.model else base / info.get("recommended_model", "battery_int8.tflite")
    labels_path = Path(a.labels) if a.labels else model.parent / "labels.txt"
    labels = [l.strip() for l in open(labels_path, encoding="utf-8") if l.strip()]
    threshold = a.threshold if a.threshold is not None else info.get("thresholds", {}).get(model.name, 0.5)

    it = Interpreter(model_path=str(model), num_threads=a.threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    size = int(inp["shape"][1])
    x = load_image(a.image, size)[None]
    if inp["dtype"] == np.uint8:
        x = x.astype(np.uint8)
    elif inp["dtype"] == np.int8:   # not produced by this pipeline, handled for safety
        s, z = inp["quantization"]
        x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
    else:
        x = x.astype(np.float32)

    t0 = time.perf_counter()
    it.set_tensor(inp["index"], x)
    it.invoke()
    probs = it.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] != np.float32:
        s, z = out["quantization"]
        probs = (probs - z) * s
    ms = (time.perf_counter() - t0) * 1000

    k = int(np.argmax(probs))
    conf = float(probs[k])
    uncertain = conf < threshold
    result = {"class": labels[k], "confidence": round(conf, 4), "uncertain": uncertain, "threshold": threshold,
              "probabilities": {l: round(float(p), 4) for l, p in zip(labels, probs)},
              "model": model.name, "inference_ms": round(ms, 1)}
    if labels[k] == "button_cell" and not info.get("button_cell_reliable", True):
        result["warning"] = "button_cell was trained without real device photos; treat as low-reliability"
    if a.json:
        print(json.dumps(result))
    else:
        tag = "UNCERTAIN" if uncertain else "OK"
        print(f"{labels[k]} {conf:.3f} [{tag}, threshold {threshold:.2f}, {ms:.0f} ms]")
        if "warning" in result:
            print("warning:", result["warning"])


if __name__ == "__main__":
    main()
