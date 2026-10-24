"""One-off: apply the visual review of the Wikimedia button-cell harvest.

Indices refer to the review contact sheets, which listed SOURCES.csv 'file' values in Python sorted() order
(sheet A = first 93, sheet B = the rest). The result is stored by Commons title in EXCLUDED_web.csv, so later
runs never depend on these indices. Rejected files are moved to downloads/wikimedia_excluded/.
"""
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config as C

DEV = "device / circuit board / holder, battery not the main subject"
PACK = "retail packaging / blister card"
MIX = "many mixed batteries / comparison collage"
DIS = "disassembled cell, device parts or unclear subject"
A = {**{i: DEV for i in [9, 11, 12, 13, 14, 23, 40, 41, 50, 59, 60, 62, 66, 68, 69, 70, 71, 72, 75, 81, 84, 88, 89, 90]},
     **{i: PACK for i in [0, 6, 10, 17, 18, 33, 38, 77]},
     **{i: MIX for i in [4, 19, 20, 37, 87]},
     **{i: DIS for i in [3, 24, 49, 57, 58, 92]}}
B = {**{i: DEV for i in [2, 3, 23, 24, 31, 32, 33, 57, 58, 61, 63, 64, 73, 79, 80, 81, 82, 83, 87, 91]},
     **{i: PACK for i in [29, 30, 38, 44, 53, 65]},
     **{i: DIS for i in [0, 1, 18, 45, 66, 85]}}
REVIEW = {**A, **{93 + i: r for i, r in B.items()}}

bc = C.PUBLIC / "button_cell"
if (bc / "EXCLUDED_web.csv").exists():
    sys.exit("Review already applied (indices are only valid for the original harvest). Nothing to do.")
rows = list(csv.DictReader(open(bc / "SOURCES.csv", encoding="utf-8-sig")))
order = sorted(r["file"] for r in rows)
by_file = {r["file"]: r for r in rows}
dest = C.DOWNLOADS / "wikimedia_excluded"
dest.mkdir(parents=True, exist_ok=True)
excl = []
for idx, reason in sorted(REVIEW.items()):
    r = by_file.pop(order[idx])
    excl.append({"commons_title": r["commons_title"], "file": r["file"], "license": r["license"], "reason": reason})
    shutil.move(str(bc / r["file"]), dest / Path(r["file"]).name)
with open(bc / "EXCLUDED_web.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, ["commons_title", "file", "license", "reason"])
    w.writeheader()
    w.writerows(excl)
with open(bc / "SOURCES.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, list(rows[0].keys()))
    w.writeheader()
    w.writerows(sorted(by_file.values(), key=lambda r: r["file"]))
print(f"excluded {len(REVIEW)}; kept {len(by_file)}")
