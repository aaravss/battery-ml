"""Step 4 (EXPERIMENT): synthetic button-cell composites. TRAIN ONLY: never written into val/test/web_holdout.

Cut-outs come only from web button-cell images assigned to the *train* split (so the web_holdout stays unseen).
A cut-out is accepted only when the source photo has a near-uniform background and the foreground mask passes
shape checks; everything else is skipped rather than producing a bad mask.
Backgrounds: battery-free patches of the RecyBat24 tray (closest thing to a device tray we have) plus plain/noisy
colours. Output: work/synthetic/button_cell/*.jpg and work/synthetic.csv (split=train, source=synthetic).
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import config as C

S = C.CROP_SAVE_SIZE


def extract_cutout(path):
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        if np.asarray(rgba)[..., 3].min() < 250:      # real transparency: use it directly
            im = rgba
            im.thumbnail((512, 512))
            a = np.asarray(im)
            mask = a[..., 3] > 128
            return finish(Image.fromarray(a[..., :3]), mask, "alpha")
        im = rgba
    im = im.convert("RGB")
    im.thumbnail((512, 512))
    a = np.asarray(im).astype(np.float32)
    b = 6
    border = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    med = np.median(border, 0)
    spread = np.median(np.linalg.norm(border - med, axis=1))
    if spread > 12:                       # cluttered / textured background: mask would be unreliable
        return None, f"background not uniform (spread {spread:.1f})"
    dist = np.linalg.norm(a - med, axis=2)
    mask = dist > max(28.0, 4 * spread)
    return finish(im, mask, "bg-threshold")


def finish(im, mask, how):
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None, "empty mask"
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    mask = lab == k
    h, w = mask.shape
    frac = mask.sum() / (h * w)
    if not 0.03 < frac < 0.85:
        return None, f"area fraction {frac:.2f}"
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()
    if edge > 0.05:
        return None, f"touches border ({edge:.2f})"
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bh, bw = y1 - y0, x1 - x0
    fill = mask[y0:y1, x0:x1].mean()
    if not (0.4 < bw / bh < 2.5 and fill > 0.6):
        return None, f"shape check failed (aspect {bw / bh:.2f}, fill {fill:.2f})"
    alpha = Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    rgba = im.convert("RGB").crop((x0, y0, x1, y1)).convert("RGBA")
    rgba.putalpha(alpha)
    return rgba, how


def tray_backgrounds(rng, n):
    root = C.DOWNLOADS / "recybat24" / "x" / "recybat24"
    out = []
    for split in ("train", "val"):
        ann = json.load(open(root / split / "annotations.json", encoding="utf-8"))
        boxes = {}
        for a in ann["annotations"]:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])
        imgs = ann["images"]
        for idx in rng.permutation(len(imgs)):
            if len(out) >= n:
                return out
            info = imgs[idx]
            side = int(rng.integers(160, 320))
            for _ in range(10):
                x, y = int(rng.integers(0, 640 - side)), int(rng.integers(0, 640 - side))
                if all(x + side < bx or bx + bw < x or y + side < by or by + bh < y
                       for bx, by, bw, bh in boxes.get(info["id"], [])):
                    im = Image.open(root / split / info["file_name"]).convert("RGB")
                    out.append(im.crop((x, y, x + side, y + side)).resize((S, S), Image.Resampling.BILINEAR))
                    break
    return out


def plain_background(rng):
    base = rng.integers(30, 230, size=3)
    grad = np.linspace(-1, 1, S)[None, :, None] * rng.normal(0, 20, size=3)
    noise = rng.normal(0, rng.uniform(2, 12), size=(S, S, 3))
    a = np.clip(base + grad + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))


def composite(cut, bg, rng):
    bg = bg.copy()
    scale = rng.uniform(0.25, 0.65) * S / max(cut.size)
    c = cut.resize((max(8, int(cut.width * scale)), max(8, int(cut.height * scale))), Image.Resampling.LANCZOS)
    c = c.rotate(rng.uniform(0, 360), expand=True, resample=Image.Resampling.BICUBIC)
    # light the cut-out roughly like the background
    rgb, al = c.convert("RGB"), c.getchannel("A")
    gain = np.clip((np.asarray(bg).mean() / max(np.asarray(rgb)[np.asarray(al) > 128].mean(), 1)) ** 0.3, 0.7, 1.3)
    rgb = Image.fromarray(np.clip(np.asarray(rgb) * gain * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8))
    x = int(rng.integers(0, max(1, S - c.width)))
    y = int(rng.integers(0, max(1, S - c.height)))
    # soft drop shadow
    sh_off = rng.integers(2, 9, size=2)
    shadow = al.filter(ImageFilter.GaussianBlur(rng.uniform(3, 7))).point(lambda v: int(v * rng.uniform(0.25, 0.45)))
    dark = Image.new("RGB", c.size, (0, 0, 0))
    bg.paste(dark, (x + int(sh_off[0]), y + int(sh_off[1])), shadow)
    bg.paste(rgb, (x, y), al)
    if rng.random() < 0.5:
        bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    return bg


def main():
    out_dir = C.WORK / "synthetic"
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "button_cell").mkdir(parents=True)
    (out_dir / "cutouts").mkdir()
    if C.SYNTHETIC_MODE == "off":
        open(C.WORK / "synthetic.csv", "w").write("path,cls,origin,source,group,merged_group,split,used\n")
        print("[synth] disabled")
        return
    rng = np.random.default_rng(C.SEED)
    rows = list(csv.DictReader(open(C.WORK / "splits.csv", encoding="utf-8")))
    src = [r for r in rows if r["cls"] == "button_cell" and r["source"] == "web" and r["split"] == "train"]
    cutouts, log = [], []
    for r in src:
        cut, how = extract_cutout(C.ROOT / r["path"])
        log.append((r["path"], "ok:" + how if cut else "skip:" + how))
        if cut is not None:
            name = Path(r["path"]).stem + ".png"
            cut.save(out_dir / "cutouts" / name)
            cutouts.append((cut, r))
    n = len(cutouts) * C.SYNTHETIC_PER_CUTOUT
    trays = tray_backgrounds(rng, int(n * 0.7) + 1)
    out_rows = []
    k = 0
    for cut, r in cutouts:
        for _ in range(C.SYNTHETIC_PER_CUTOUT):
            bg = trays[k % len(trays)] if (rng.random() < 0.7 and trays) else plain_background(rng)
            img = composite(cut, bg, rng)
            p = out_dir / "button_cell" / f"syn_{k:05d}.jpg"
            img.save(p, quality=int(rng.integers(85, 96)))
            out_rows.append(dict(path=str(p.relative_to(C.ROOT)).replace("\\", "/"), cls="button_cell",
                                 origin="synthetic", source="synthetic", group=r["merged_group"],
                                 merged_group=r["merged_group"], split="train", used=1))
            k += 1
    with open(C.WORK / "synthetic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(out_rows[0].keys()) if out_rows else ["path"])
        w.writeheader()
        w.writerows(out_rows)
    with open(out_dir / "cutout_log.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("source_image", "result"), *log])
    print(f"[synth] {len(cutouts)}/{len(src)} train web images gave usable cut-outs -> {len(out_rows)} synthetic images")


if __name__ == "__main__":
    main()
