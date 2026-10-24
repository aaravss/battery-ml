"""Save the trained model as one TensorFlow/Keras file and make a small zip to send to a mentor.

    .venv\\Scripts\\python.exe tools\\export_keras.py      (or double-click export_keras.bat)

No retraining: it rebuilds the network from pipeline/model_def.py, loads runs/<variant>/best.weights.h5,
saves models/battery_model.keras, reloads it and checks it gives the same answers as battery_float32.tflite.
Then it writes battery-model-for-mentor.zip (model + code + report, no datasets, no .venv).
"""
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
LOG = ROOT / "logs" / "export_keras.log"
LOG.parent.mkdir(exist_ok=True)
log_lines = []


def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))
    LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


HOW_TO = """Battery shape classifier (MobileNetV2, Keras 3 / TensorFlow)

Files
  battery_model.keras      full Keras model (architecture + trained weights)
  battery_float32.tflite   same model as TFLite
  battery_int8.tflite      quantised TFLite for the Raspberry Pi (2.7 MB)
  labels.txt               class order of the 4 outputs
  model_info.json          image size, classes, confidence threshold, preprocessing
  infer_pi.py              inference script for the Pi
  REPORT.md                full write-up: data, training, results, known weaknesses
  pipeline/                training code (run_pipeline.bat re-downloads data and retrains)

Load in Python (TensorFlow 2.21 / Keras 3.15):
  import keras, numpy as np
  model = keras.models.load_model("battery_model.keras")
  # input: (N, 224, 224, 3) float32 RGB in 0-255 (scaling is inside the model)
  probs = model.predict(images)          # 4 softmax outputs, order as labels.txt
  # treat max prob < 0.70 as "uncertain" (see model_info.json)

Caveat (see REPORT.md): 99.3% test accuracy is on public photos only; the model has
not yet been tested on photos from the real device camera, and button_cell is unreliable.
"""


def main():
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import keras
    from model_def import build_model

    info = json.loads((ROOT / "models" / "model_info.json").read_text(encoding="utf-8"))
    variant, size = info.get("variant", "baseline"), info["img_size"]
    log(f"TF {tf.__version__}, Keras {keras.__version__}; variant={variant}, img_size={size}")

    model, _ = build_model(size, weights=None)
    model.load_weights(ROOT / "runs" / variant / "best.weights.h5")
    out = ROOT / "models" / "battery_model.keras"
    model.save(out)
    log(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # verify: reload the .keras file and compare with the shipped float32 TFLite model
    reloaded = keras.models.load_model(out)
    X = np.random.default_rng(0).uniform(0, 255, (8, size, size, 3)).astype(np.float32)
    pk = reloaded.predict(X, verbose=0)
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(ROOT / "models" / "battery_float32.tflite"))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    pt = []
    for x in X:
        it.set_tensor(i["index"], x[None])
        it.invoke()
        pt.append(it.get_tensor(o["index"])[0])
    diff = float(np.abs(pk - np.array(pt)).max())
    log(f"reloaded .keras vs battery_float32.tflite: max |diff| = {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("MISMATCH: the .keras file does not match the shipped model")
    log("VERIFIED: battery_model.keras gives the same outputs as the shipped model")

    # small zip for a mentor: no datasets, no .venv/.cache
    zpath = ROOT / "battery-model-for-mentor.zip"
    files = [p for p in (ROOT / "models").iterdir() if p.is_file()]
    files += sorted((ROOT / "pipeline").glob("*.py"))
    files += [p for p in (ROOT / "reports").rglob("*") if p.is_file()] if (ROOT / "reports").exists() else []
    files += [ROOT / n for n in ("REPORT.md", "requirements.txt", "run_pipeline.bat", "run_pipeline.ps1")
              if (ROOT / n).exists()]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("HOW_TO_LOAD.txt", HOW_TO)
    log(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB, {len(files) + 1} files)")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        log(f"FAILED: {e}")
        sys.exit(1)
    except Exception:
        log("FAILED:\n" + traceback.format_exc())
        sys.exit(1)
