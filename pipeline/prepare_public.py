"""Step 2: turn downloaded datasets into data/public/<class>/<source>/ images.

RecyBat24  -> one square crop per bounding box (CROP_MARGIN x longest side), so a crop looks like a single battery
              in a tray, closer to the device camera than the 640x640 overview frames. Filename prefix before "__"
              is the physical-battery group used for leak-free splitting.
Singapore  -> padded to square with the image's own border colour (no black bars that a model could key on).
Web button cells are already in data/public/button_cell/web/ (fetch_public.py) and are left untouched.
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from imgutil import pad_to_square


def recybat():
    root = C.DOWNLOADS / "recybat24" / "x" / "recybat24"
    kept, dropped = Counter(), Counter()
    for c in C.CLASSES:
        shutil.rmtree(C.PUBLIC / c / "recybat24", ignore_errors=True)
    for split in ("train", "val"):   # the dataset's own split is ignored: same physical batteries appear in both
        ann = json.load(open(root / split / "annotations.json", encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in ann["categories"]}
        imgs = {i["id"]: i for i in ann["images"]}
        for a in ann["annotations"]:
            name = cats[a["category_id"]]
            cls = C.RECYBAT_MAP[name]
            if name == "prismatic" and C.RECYBAT_PRISMATIC_TO_POUCH:
                cls = "phone_pouch"
            if cls is None:
                dropped[name] += 1
                continue
            info = imgs[a["image_id"]]
            stem = Path(info["file_name"]).stem
            battery_id = stem.split("_")[0]
            im = Image.open(root / split / info["file_name"]).convert("RGB")
            x, y, w, h = a["bbox"]
            side = min(max(w, h) * C.CROP_MARGIN, min(im.size))
            cx, cy = x + w / 2, y + h / 2
            left = int(np.clip(cx - side / 2, 0, im.width - side))
            top = int(np.clip(cy - side / 2, 0, im.height - side))
            crop = im.crop((left, top, left + int(side), top + int(side))).resize(
                (C.CROP_SAVE_SIZE, C.CROP_SAVE_SIZE), Image.Resampling.LANCZOS)
            out = C.PUBLIC / cls / "recybat24" / f"rb_{name[:3]}{battery_id}__{stem}_a{a['id']}_{split}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out, quality=92)
            kept[cls] += 1
    return {"kept": dict(kept), "dropped": dict(dropped)}


def singapore():
    if not C.USE_SINGAPORE:
        for c in C.CLASSES:
            shutil.rmtree(C.PUBLIC / c / "singapore", ignore_errors=True)
        return {"disabled": True}
    root = C.DOWNLOADS / "singapore" / "x" / "Singapore_Battery_Dataset-main"
    kept, dropped, unreadable = Counter(), [], []
    for c in C.CLASSES:
        shutil.rmtree(C.PUBLIC / c / "singapore", ignore_errors=True)
    for folder, cls in C.SINGAPORE_MAP.items():
        for f in sorted((root / folder).iterdir()):
            if f.suffix.lower() not in C.IMG_EXTS:
                continue
            rel = f"{folder}/{f.name}"
            if rel in C.SINGAPORE_EXCLUDE:
                dropped.append(rel)
                continue
            try:
                im = Image.open(f)
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    im = Image.alpha_composite(bg, im)
                im = pad_to_square(im.convert("RGB"))
                im.thumbnail((512, 512), Image.Resampling.LANCZOS)
            except Exception as e:
                unreadable.append(f"{rel}: {e}")
                continue
            out = C.PUBLIC / cls / "singapore" / f"sg_{folder}_{f.stem.replace(' ', '_')}_{f.suffix[1:].lower()}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, quality=92)
            kept[cls] += 1
    return {"kept": dict(kept), "dropped_manual_review": dropped, "unreadable": unreadable}


if __name__ == "__main__":
    C.WORK.mkdir(exist_ok=True)
    stats = {}
    for name, fn in [("recybat24", recybat), ("singapore", singapore)]:
        try:
            stats[name] = fn()
        except FileNotFoundError as e:
            print(f"  WARNING: {name} not available ({e}); continuing without it")
            stats[name] = {"missing": str(e)}
        print(f"[prepare] {name}: {json.dumps(stats[name])[:400]}")
    json.dump(stats, open(C.WORK / "prepare_stats.json", "w"), indent=2)
