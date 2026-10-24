"""Step 1: download public data (idempotent: anything already on disk is skipped, so offline re-runs work).

- RecyBat24 (Zenodo 15226091, CC-BY 4.0)      -> downloads/recybat24/
- Singapore Battery Dataset (GitHub, no license) -> downloads/singapore/
- Wikimedia Commons button/coin cells (per-file CC0/CC-BY/CC-BY-SA/PD) -> data/public/button_cell/web/ + SOURCES.csv
"""
import csv
import html
import re
import sys
import tarfile
import time
import zipfile
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import config as C

UA = {"User-Agent": "battery-shape-classifier-student-project/1.0 (educational e-waste safety project; python-requests)"}
WIKI_API = "https://commons.wikimedia.org/w/api.php"
SOURCES_FIELDS = ["file", "source_url", "commons_title", "author", "license", "license_url", "categories", "downloaded"]


def download(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def safe_extract_tar(archive: Path, out: Path):
    with tarfile.open(archive) as t:
        t.extractall(out, filter="data")   # 'data' filter blocks absolute paths / traversal / links outside out


def safe_extract_zip(archive: Path, out: Path):
    out_res = out.resolve()
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            if not (out / m.filename).resolve().is_relative_to(out_res):
                raise RuntimeError(f"unsafe path in zip: {m.filename}")
        z.extractall(out)


def fetch_recybat():
    d = C.DOWNLOADS / "recybat24"
    download("https://zenodo.org/records/15226091/files/recybat24.tar.gz?download=1", d / "recybat24.tar.gz")
    if not (d / "x" / "recybat24" / "train" / "annotations.json").exists():
        print("  extracting RecyBat24")
        safe_extract_tar(d / "recybat24.tar.gz", d / "x")


def fetch_singapore():
    d = C.DOWNLOADS / "singapore"
    download("https://codeload.github.com/FriedrichZhao/Singapore_Battery_Dataset/zip/refs/heads/main", d / "repo.zip")
    if not (d / "x" / "Singapore_Battery_Dataset-main").exists():
        print("  extracting Singapore")
        safe_extract_zip(d / "repo.zip", d / "x")


# ---------------------------------------------------------------- Wikimedia --
def wiki_get(params):
    params = {**params, "format": "json", "formatversion": "2"}
    for attempt in range(5):
        try:
            r = requests.get(WIKI_API, params=params, headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print("  retry", e)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Wikimedia API unreachable")


def walk_category(root):
    """Yield (file_title, category) for files under root, recursing into subcategories."""
    seen, stack = set(), [(root, 0)]
    while stack:
        cat, depth = stack.pop()
        if cat in seen or cat in C.WIKI_SKIP_CATEGORIES:
            continue
        seen.add(cat)
        cont = {}
        while True:
            j = wiki_get({"action": "query", "list": "categorymembers", "cmtitle": cat,
                          "cmtype": "file|subcat", "cmlimit": "500", **cont})
            for m in j["query"]["categorymembers"]:
                if m["ns"] == 14 and depth < C.WIKI_MAX_DEPTH:
                    stack.append((m["title"], depth + 1))
                elif m["ns"] == 6:
                    yield m["title"], cat
            if "continue" not in j:
                break
            cont = j["continue"]


def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def license_ok(code, short):
    s = f"{code} {short}".lower()
    if "nc" in re.split(r"[-\s]", s) or "nd" in re.split(r"[-\s]", s):
        return False
    return any(s.startswith(p) or f" {p}" in s for p in C.ALLOWED_LICENSE_PREFIXES)


def safe_name(title):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", title.removeprefix("File:"))
    return stem[:120]


def fetch_wikimedia(refresh=False):
    out = C.PUBLIC / "button_cell" / "web"
    out.mkdir(parents=True, exist_ok=True)
    src_csv, excl_csv = out.parent / "SOURCES.csv", out.parent / "EXCLUDED_web.csv"
    if src_csv.exists() and not refresh:
        # New Commons uploads would enter training without a visual review, so harvesting is opt-in after the first run.
        print("  SOURCES.csv exists -> skipping harvest (use --refresh-web to look for new files, then review them)")
        return
    rows = {r["commons_title"]: r for r in csv.DictReader(open(src_csv, encoding="utf-8-sig"))} if src_csv.exists() else {}
    excluded = {r["commons_title"] for r in csv.DictReader(open(excl_csv, encoding="utf-8-sig"))} if excl_csv.exists() else set()

    files = {}
    for root in C.WIKI_ROOT_CATEGORIES:
        for title, cat in walk_category(root):
            files.setdefault(title, set()).add(cat)
    print(f"  Wikimedia: {len(files)} candidate files, {len(rows)} already kept, {len(excluded)} excluded by review")

    todo = [t for t in files if t not in rows and t not in excluded]
    rejected = []
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        j = wiki_get({"action": "query", "titles": "|".join(batch), "prop": "imageinfo",
                      "iiprop": "url|extmetadata|mime|size", "iiurlwidth": str(C.WIKI_THUMB_WIDTH)})
        for p in j["query"]["pages"]:
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            meta = ii.get("extmetadata", {})
            code = meta.get("License", {}).get("value", "")
            short = meta.get("LicenseShortName", {}).get("value", "")
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                rejected.append((p["title"], f"mime {ii.get('mime')}"))
                continue
            if not license_ok(code, short):
                rejected.append((p["title"], f"license '{short or code}' not in allowed list"))
                continue
            ext = ".png" if ii["mime"] == "image/png" else ".jpg"
            fname = safe_name(p["title"]).rsplit(".", 1)[0] + ext
            try:
                download(ii.get("thumburl") or ii["url"], out / fname)
            except requests.RequestException as e:
                rejected.append((p["title"], f"download failed: {e}"))
                continue
            rows[p["title"]] = {
                "file": f"web/{fname}", "source_url": ii["descriptionurl"], "commons_title": p["title"],
                "author": strip_html(meta.get("Artist", {}).get("value", ""))[:200],
                "license": short or code, "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "categories": ";".join(sorted(files[p["title"]])), "downloaded": str(date.today()),
            }
            time.sleep(0.2)
    with open(src_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SOURCES_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: r["file"]))
    with open(C.DOWNLOADS / "wikimedia_rejected.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("commons_title", "reason"), *rejected])
    print(f"  Wikimedia: kept {len(rows)} files; rejected {len(rejected)} this run (license/mime)")


if __name__ == "__main__":
    refresh = "--refresh-web" in sys.argv
    for name, fn in [("RecyBat24", fetch_recybat), ("Singapore", fetch_singapore),
                     ("Wikimedia", lambda: fetch_wikimedia(refresh))]:
        print(f"[fetch] {name}")
        try:
            fn()
        except Exception as e:   # a dead mirror must not block training on what is already on disk
            print(f"  WARNING: {name} fetch failed: {e!r}; continuing with whatever is on disk")
