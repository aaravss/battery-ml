"""Central configuration for the battery-shape pipeline. Every decision that a re-run depends on lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PUBLIC = DATA / "public"
OWN = DATA / "own"
DOWNLOADS = ROOT / "downloads"
WORK = ROOT / "work"            # derived, regenerated every run (splits, synthetic images, caches)
RUNS = ROOT / "runs"            # checkpoints + logs per training variant
MODELS = ROOT / "models"        # final deliverables (tflite, labels.txt, threshold.json)
REPORTS = ROOT / "reports"      # auto-generated metrics, plots, RESULTS.md

CLASSES = ["button_cell", "aa_aaa_9v", "cylindrical_18650", "phone_pouch"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---- model / training -------------------------------------------------------
IMG_SIZE = 224                  # GPU available -> full MobileNetV2 resolution (CPU fallback would use 160)
IMG_SIZE_CPU = 160
BATCH = 32
HEAD_EPOCHS = 15
FT_EPOCHS = 25
HEAD_LR = 1e-3
FT_LR = 1e-5
FT_UNFREEZE_FROM = "block_13_expand"   # unfreeze from here to the top (~last 3 inverted-residual blocks + head conv)
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
SEED = 1337

# ---- splitting --------------------------------------------------------------
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / val / test, assigned per GROUP, never per image
PHASH_MAX_DIST = 4                    # perceptual-hash Hamming distance that counts as "near duplicate"
TRAIN_CAP_PER_GROUP = 12               # max train images from one physical battery / photo series (de-redundancy)
CLASS_WEIGHT_CLIP = (0.5, 5.0)

# Web-sourced button cells: train + a separate UNRELIABLE holdout. Never val/test (user rule 5e).
WEB_BUTTON_HOLDOUT_FRACTION = 0.20

# ---- sources ----------------------------------------------------------------
USE_SINGAPORE = True            # no license file + images scraped from Google/retailers (see REPORT.md); set False to drop
RECYBAT_PRISMATIC_TO_POUCH = False   # user said drop prismatic; flip to True to treat flat prismatic cells as phone_pouch
CROP_MARGIN = 1.6               # RecyBat24 crops: square side = CROP_MARGIN * longest bbox side
CROP_SAVE_SIZE = 320

# Synthetic button-cell compositing: "off" | "experiment" (train both, keep better per rule) | "on"
SYNTHETIC_MODE = "experiment"
SYNTHETIC_PER_CUTOUT = 8
SYNTH_KEEP_RULE = dict(min_button_recall_gain=0.05, max_val_acc_drop=0.01)

# int8 is shipped as the recommended model only if it stays within this accuracy of float32 on the test set
INT8_MAX_ACC_DROP = 0.01

# Threshold calibration: pick the lowest threshold whose selective accuracy on validation >= this target
THRESHOLD_TARGET_ACC = 0.98
# Lower bound on the shipped threshold. The public validation set is saturated (100% accurate above 0.5), so it
# cannot calibrate anything; on the shortcut diagnostic (domain-shifted images) 0.70 halved confident errors
# while keeping ~72% coverage (see REPORT.md section 9). Revisit once data/own/ feeds validation.
THRESHOLD_FLOOR = 0.70

# ---- manual exclusions (visual review, documented in REPORT.md) -------------
# Singapore dataset: files whose battery is NOT the shape of the folder's mapped class.
SINGAPORE_EXCLUDE = {
    # 'Other' folder: Panasonic 18650-type Li-ion, 2/3A lithium, unclear small green cells
    "Other/11.jpg", "Other/13.jpg", "Other/15.jpg", "Other/17 2.jpg", "Other/9.jpg", "Other/test_battery_back.jpg",
    # Exell_Green: look like 3.7 V 18650 Li-ion rather than AA NiMH
    "Exell_Green/28.jpg", "Exell_Green/29.jpg", "Exell_Green/31.jpg",
    # Klarus: 14500 cells (AA-sized Li-ion, 750/800 mAh) -> not 18650-shaped
    "Klarus/28.jpg", "Klarus/36.jpg", "Klarus/37.jpg", "Klarus/38.jpg", "Klarus/40.jpg", "Klarus/41.jpg",
    "Klarus/44.jpg", "Klarus/45.jpg",
}
SINGAPORE_MAP = {
    "Duracell_Alk": "aa_aaa_9v", "Energizer_Alk": "aa_aaa_9v", "Energizer_Ind": "aa_aaa_9v",
    "Energizer_Li": "aa_aaa_9v", "Exell_Green": "aa_aaa_9v", "Exell_Yellow": "aa_aaa_9v",
    "GP_Alk": "aa_aaa_9v", "IKEA_Alk": "aa_aaa_9v", "Other": "aa_aaa_9v",
    "Klarus": "cylindrical_18650",
}
RECYBAT_MAP = {"cylindric": "cylindrical_18650", "pouch": "phone_pouch", "prismatic": None}

# Wikimedia Commons button-cell harvest
WIKI_ROOT_CATEGORIES = ["Category:Button cells", "Category:Coin cells"]
WIKI_SKIP_CATEGORIES = {
    "Category:Button cells in the battery compartment of devices",
    "Category:Button cells mounted/fixed to a printed circuit board of a device",
}
WIKI_MAX_DEPTH = 3
ALLOWED_LICENSE_PREFIXES = ("cc0", "cc-by-", "cc by ", "cc-by-sa", "cc by-sa", "public domain", "pd")
WIKI_THUMB_WIDTH = 640
# Files removed after visual review (diagrams, devices, packaging, X-rays, ...): see data/public/button_cell/web/EXCLUDED.csv
