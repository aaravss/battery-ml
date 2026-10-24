# battery-ml: battery shape classifier for e-waste safety

> **This repository must stay private.** It contains the Singapore Battery Dataset images, which have no
> license (see [Data sources and licenses](#data-sources-and-licenses)).

## What this is

A student project: a small device that helps informal e-waste workers identify a battery **before** they decide
whether to open it. A fixed camera (~70 mm above a tray) photographs one battery. This model classifies the
battery's **shape**. A separate rule-based layer (not in this repo) combines that shape with a load-cell weight
reading to decide *safe* vs *hazardous*. The model runs offline on a **Raspberry Pi 4 (Debian 13)** as TensorFlow
Lite.

### The 4 classes

| class | what it covers |
|---|---|
| `button_cell` | coin and button cells (CR2032, LR44, SR626, zinc-air, ...) |
| `aa_aaa_9v` | alkaline-shaped cells: AA, AAA, C, D, 9 V (also NiMH/lithium AA) |
| `cylindrical_18650` | cylindrical Li-ion cells (18650, some 21700), including small welded packs |
| `phone_pouch` | flat phone/tablet pouch Li-ion batteries |

Any prediction with confidence below **0.70** is reported as *uncertain*, and the device should treat it as
unidentified.

## Training setup

* **Model:** MobileNetV2 with ImageNet weights (transfer learning), 224×224 RGB input, then global average
  pooling, Dropout 0.3, and Dense(4, softmax). Pixel scaling is inside the model, so it takes raw 0–255 RGB.
* **Phase 1, 15 epochs:** backbone frozen, new head trained with Adam lr 1e-3.
* **Phase 2, 25 epochs:** fine-tune the top of the backbone (from `block_13_expand`, 26 layers; BatchNorm kept
  frozen) with Adam lr **1e-5**.
* Both phases use early stopping on validation loss (best weights restored); phase 2 also uses ReduceLROnPlateau.
  Label smoothing is 0.05.
* **Augmentation (training images only, never validation/test):** random rotation (full 360°), shift ±10%,
  zoom ±15%, horizontal and vertical flips, brightness ±20%, contrast 0.7–1.3, mild saturation 0.8–1.2, and
  Gaussian blur on 30% of images.
* **Class imbalance:** at most 12 training images per physical battery (RecyBat24 shoots each battery ~30 times),
  plus class weights (clipped to 0.5–5).
* **Leakage control:** train/val/test (70/15/15) is split **by physical battery / photo group before any
  augmentation**, and near-duplicate images are merged by perceptual hash so they land in the same split.
* Trained with Keras 3 on an RTX 5070 (PyTorch backend); exported to TFLite with TensorFlow 2.21.

Full details and every decision: [REPORT.md](REPORT.md).

## Results

Held-out test set (public images only; shipped model `battery_int8.tflite`):

| class | precision | recall | F1 | test images |
|---|---|---|---|---|
| `button_cell` | – | – | – | 0 (no real device photos yet, see below) |
| `aa_aaa_9v` | 0.970 | 0.970 | 0.970 | 33 |
| `cylindrical_18650` | 0.988 | 0.988 | 0.988 | 85 |
| `phone_pouch` | 1.000 | 1.000 | 1.000 | 180 |
| **overall accuracy** | | | **99.3%** | 298 |

* **int8 vs float32:** same test accuracy (99.3%); the int8 model is 2.7 MB vs 8.7 MB, so int8 is the default.
* **At the 0.70 threshold:** 99.0% of test images are accepted, and all accepted ones are correct.
* **button_cell:** 24/24 held-out Wikimedia photos were correct, but this is **unreliable**: there are no
  photos from the device camera, so button_cell is not in the test set.

**Important caveats.** These numbers are a **baseline on public data**, not a prediction for the device:

* No training image came from the device camera, tray, lighting or 70 mm distance.
* Each class mostly comes from a single source (AA from product photos, pouch from one lab tray), so the model
  partly learned the background. In a diagnostic where held-out AA cells were pasted onto the lab tray, only
  **78%** were recognised (88.9% overall on that diagnostic).
* The test set is small and clustered: the 180 pouch images come from only 6 physical batteries.

The next step is photographing real batteries on the device (REPORT.md §13 has the shot list).

Plots and tables: [reports/RESULTS.md](reports/RESULTS.md),
[confusion matrix](reports/confusion_matrix_test.png), [threshold trade-off](reports/threshold_tradeoff.png),
[training curves](reports/training_curves.png).

## Data sources and licenses

| source | used for | license |
|---|---|---|
| **RecyBat24** (Acaro Chacón et al., *Scientific Data* 12, 843, 2025; Zenodo 15226091) | `cylindrical_18650`, `phone_pouch` (cropped from bounding boxes; prismatic cells dropped) | CC BY 4.0, attribution required |
| **Singapore Battery Dataset / BatSort** (Zhao et al., arXiv 2404.05802) | `aa_aaa_9v`, some `cylindrical_18650` | **No license.** The images were collected by the authors from Google Images and shopping sites, so copyright stays with the original owners. **Do not make this repository public or redistribute these images.** |
| **Wikimedia Commons** (button/coin cell categories, 110 photos after manual review) | `button_cell` | Per image: CC0 / public domain / CC BY / CC BY-SA. Author, license and URL of every image are in [data/public/button_cell/SOURCES.csv](data/public/button_cell/SOURCES.csv). |

The device's own photos will go in `data/own/<class>/<battery_id>/` (see [data/own/README.md](data/own/README.md)).

## Repository layout

| path | contents |
|---|---|
| `models/` | `battery_int8.tflite` (recommended), `battery_float32.tflite`, `battery_model.keras`, `labels.txt`, `model_info.json`, `infer_pi.py` |
| `data/public/` | the prepared training images per class and source |
| `pipeline/` | data prep → split → train → export → evaluate (settings in `pipeline/config.py`) |
| `reports/` | metrics, confusion matrices, plots |
| `REPORT.md` | full technical report, including weaknesses |
| `infer_pi.py` | single-image inference for the Raspberry Pi |
| `run_pipeline.bat` / `run_pipeline.ps1` | the one-command pipeline (Windows) |

## How to load the model

**TFLite on a Raspberry Pi (64-bit Debian 13 / Python 3.13).** `tflite-runtime` has no wheels for Python 3.13,
so use `ai-edge-litert`:

```bash
python3 -m venv ~/battery-venv
~/battery-venv/bin/pip install ai-edge-litert pillow
~/battery-venv/bin/python models/infer_pi.py photo.jpg          # e.g. "phone_pouch 0.996 [OK, threshold 0.70, ...]"
~/battery-venv/bin/python models/infer_pi.py photo.jpg --json   # JSON line for the rule layer
```

**TFLite in your own Python code:**

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

labels = open("models/labels.txt").read().split()
it = Interpreter(model_path="models/battery_int8.tflite")
it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]

img = Image.open("photo.jpg").convert("RGB").resize((224, 224))   # infer_pi.py also pads to square first
x = np.asarray(img, dtype=np.uint8)[None]                          # int8 model: uint8 0-255; float32 model: float32 0-255
it.set_tensor(inp["index"], x)
it.invoke()
probs = it.get_tensor(out["index"])[0]
k = int(probs.argmax())
print(labels[k], probs[k], "uncertain" if probs[k] < 0.70 else "")
```

**Keras (TensorFlow 2.21 / Keras 3.15):**

```python
import keras
model = keras.models.load_model("models/battery_model.keras")
probs = model.predict(images)   # images: (N, 224, 224, 3) float32 RGB 0-255; outputs follow labels.txt order
```

For identical results, preprocess the same way as training: EXIF-rotate, pad to square with the median border
colour, then resize to 224×224. This is `load_image()` in `infer_pi.py` / `pipeline/imgutil.py`.

## Re-running the pipeline (Windows)

Large and rebuildable folders are not in the repo: `.venv/`, `.cache/`, `downloads/`, `runs/`, `work/`, `logs/`
and `tools/uv/`. To retrain from a fresh clone:

1. Put a portable `uv.exe` in `tools/uv/` (from <https://github.com/astral-sh/uv/releases>,
   `uv-x86_64-pc-windows-msvc.zip`).
2. Run `run_pipeline.bat`. It creates the Python 3.12 environment, re-downloads RecyBat24 and the Singapore
   set, and runs every step.

Training takes about 5 minutes on an NVIDIA GPU, longer on CPU.
