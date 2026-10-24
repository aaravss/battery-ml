"""Step 3: assign every image to train / val / test (or web_holdout) BEFORE any augmentation, without leakage.

Grouping (images in one group always share a split):
  * data/own/<class>/<subfolder>/...   -> group = subfolder   (put each physical battery in its own subfolder)
  * filename "<group>__<rest>.jpg"     -> group = <group>      (RecyBat24 crops use the physical battery id)
  * otherwise                          -> its own group
  * then any two images whose perceptual hashes differ by <= PHASH_MAX_DIST are merged into one group
    (catches re-uploads / crops of the same photo across sources).
Split assignment is a stable hash of the merged group id, so adding new photos does not reshuffle old ones.
Web button cells never go to val/test: they go to train or to a separate 'web_holdout' (user rule 5e).
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def stable_u(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16) / float(1 << 48)


def scan():
    rows = []
    for origin, base in (("public", C.PUBLIC), ("own", C.OWN)):
        for cls in C.CLASSES:
            d = base / cls
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.suffix.lower() not in C.IMG_EXTS or not p.is_file():
                    continue
                rel = p.relative_to(d)
                if origin == "public":
                    source = rel.parts[0] if len(rel.parts) > 1 else "public_misc"
                    explicit = "__" in p.stem
                    group = p.stem.split("__")[0] if explicit else str(rel)
                else:
                    source = "own"
                    explicit = len(rel.parts) > 1 or "__" in p.stem
                    group = rel.parts[0] if len(rel.parts) > 1 else (p.stem.split("__")[0] if "__" in p.stem else str(rel))
                rows.append(dict(path=str(p.relative_to(C.ROOT)).replace("\\", "/"), cls=cls, origin=origin,
                                 source=source, group=f"{origin}/{cls}/{source}/{group}", explicit=explicit))
    return rows


def phashes(rows):
    cache_f = C.WORK / "phash_cache.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
    out = []
    for r in rows:
        p = C.ROOT / r["path"]
        key = f"{r['path']}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
        if key not in cache:
            try:
                cache[key] = str(imagehash.phash(Image.open(p).convert("RGB")))
            except Exception as e:
                print("  unreadable, skipped:", r["path"], e)
                cache[key] = None
        out.append(cache[key])
    json.dump(cache, open(cache_f, "w"))
    return out


class DSU:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def main():
    C.WORK.mkdir(exist_ok=True)
    rows = scan()
    hashes = phashes(rows)
    keep = [i for i, h in enumerate(hashes) if h is not None]
    rows = [rows[i] for i in keep]
    hashes = [hashes[i] for i in keep]

    # Near-duplicate merge (vectorised Hamming distance on 64-bit pHashes). Only pairs where at least one image
    # lacks an explicit group are merged: RecyBat24 crops share a grey tray background, and letting them chain
    # through pHash collapsed ~60 physical batteries into a handful of giant groups. Cross-class pairs are only
    # reported (they are product shots that look alike, not true duplicates).
    bits = np.array([[int(b) for b in bin(int(h, 16))[2:].zfill(64)] for h in hashes], dtype=np.uint8)
    dsu = DSU({r["group"] for r in rows})
    cross_class = []
    n_pairs = 0
    for i in range(len(rows)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(d <= C.PHASH_MAX_DIST)[0] + i + 1:
            if rows[i]["explicit"] and rows[j]["explicit"]:
                continue
            if rows[i]["cls"] != rows[j]["cls"]:
                cross_class.append((rows[i]["path"], rows[j]["path"], int(d[j - i - 1])))
                continue
            n_pairs += 1
            dsu.union(rows[i]["group"], rows[j]["group"])
    for r in rows:
        r["merged_group"] = dsu.find(r["group"])

    own_button = any(r["cls"] == "button_cell" and r["origin"] == "own" for r in rows)
    tr, va, _ = C.SPLIT_FRACTIONS
    for r in rows:
        u = stable_u(r["merged_group"])
        if r["cls"] == "button_cell" and r["source"] == "web":
            r["split"] = "train" if u >= C.WEB_BUTTON_HOLDOUT_FRACTION else "web_holdout"
        else:
            r["split"] = "train" if u < tr else ("val" if u < tr + va else "test")

    # Guarantee every class/origin with >= 3 groups has at least one group in val and in test.
    by_key = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split"] in ("train", "val", "test") and not (r["cls"] == "button_cell" and r["source"] == "web"):
            by_key[(r["cls"], r["origin"])][r["merged_group"]].append(r)
    for key, groups in by_key.items():
        if len(groups) < 3:
            continue
        for need in ("test", "val"):
            if not any(g[0]["split"] == need for g in groups.values()):
                donors = sorted((g for g in groups.values() if g[0]["split"] == "train"),
                                key=lambda g: -stable_u(g[0]["merged_group"]))
                if donors:
                    for r in donors[0]:
                        r["split"] = need

    # De-redundancy cap on train images per physical battery / series
    rng = np.random.default_rng(C.SEED)
    for r in rows:
        r["used"] = 1
    per_group = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            per_group[r["merged_group"]].append(r)
    capped = Counter()
    for g, rs in per_group.items():
        if len(rs) > C.TRAIN_CAP_PER_GROUP and rs[0]["origin"] == "public":
            for idx in rng.permutation(len(rs))[C.TRAIN_CAP_PER_GROUP:]:
                rs[idx]["used"] = 0
                capped[rs[idx]["cls"]] += 1

    fields = ["path", "cls", "origin", "source", "group", "merged_group", "split", "used"]
    with open(C.WORK / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)

    counts = defaultdict(Counter)
    for r in rows:
        if r["used"]:
            counts[f"{r['cls']}|{r['source']}"][r["split"]] += 1
    groups_per_cls = {c: len({r["merged_group"] for r in rows if r["cls"] == c}) for c in C.CLASSES}
    stats = dict(n_images=len(rows), near_dup_pairs=n_pairs, cross_class_near_dups=cross_class,
                 capped_out_of_train=dict(capped), own_button_cell_present=own_button,
                 counts={k: dict(v) for k, v in sorted(counts.items())}, groups_per_class=groups_per_cls,
                 raw_per_class=dict(Counter(r["cls"] for r in rows)))
    json.dump(stats, open(C.WORK / "split_stats.json", "w"), indent=2)
    print(f"[splits] {len(rows)} images, {n_pairs} near-duplicate pairs merged, "
          f"{len(cross_class)} cross-class near-dups, capped {dict(capped)}")
    for k, v in sorted(counts.items()):
        print(f"  {k:40s} {dict(v)}")
    for c in C.CLASSES:
        n = sum(1 for r in rows if r["cls"] == c and r["used"] and r["split"] != "web_holdout")
        if n < 100:
            print(f"  !!! WARNING: class {c} has only {n} usable real images. Results for it are not trustworthy. !!!")
    if not own_button:
        print("  !!! button_cell has NO own photos: it is absent from val/test; its metrics come only from the "
              "web_holdout and are UNRELIABLE. !!!")


if __name__ == "__main__":
    main()
