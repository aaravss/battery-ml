"""Step 7: evaluate every exported variant with the same LiteRT interpreter the Pi uses, calibrate the
"uncertain" threshold on validation, pick the model to ship, and write reports/ + models/.

Sets evaluated:
  val          - threshold calibration + variant selection (button_cell absent until own photos exist)
  test         - held-out headline metrics (never used for any choice except the int8-vs-float32 check)
  web_holdout  - web button-cell photos never trained on: button_cell metrics, labelled UNRELIABLE
  diagnostic   - SHORTCUT CHECK, not a test set: held-out product-shot AA/18650 cells and holdout button cells
                 cut out and pasted onto the RecyBat24 tray. Reported separately; never mixed into test metrics.
"""
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from sklearn.metrics import classification_report, confusion_matrix

import config as C
from imgutil import load_image
from synth_buttons import composite, extract_cutout, tray_backgrounds

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
NC = len(C.CLASSES)
CIDX = {c: i for i, c in enumerate(C.CLASSES)}


def predict(model_path, X, threads=8):
    it = Interpreter(model_path=str(model_path), num_threads=threads)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = np.zeros((len(X), NC), np.float32)
    for k, x in enumerate(X):
        it.set_tensor(i["index"], x[None].astype(i["dtype"]))
        it.invoke()
        out[k] = it.get_tensor(o["index"])[0]
    return out


def build_diagnostic(rows, size):
    """Cut-outs of held-out (test / web_holdout) uniform-background photos pasted on the tray."""
    rng = np.random.default_rng(C.SEED + 7)
    src = [r for r in rows if (r["split"] == "test" and r["source"] == "singapore")
           or r["split"] == "web_holdout"]
    trays = tray_backgrounds(rng, 200)
    X, y = [], []
    for r in src:
        cut, _ = extract_cutout(C.ROOT / r["path"])
        if cut is None:
            continue
        for _ in range(3):
            img = composite(cut, trays[int(rng.integers(len(trays)))], rng)
            X.append(np.asarray(img.resize((size, size)), dtype=np.uint8))
            y.append(CIDX[r["cls"]])
    return (np.stack(X), np.array(y)) if X else (np.zeros((0, size, size, 3), np.uint8), np.zeros(0, int))


def selective(p, y):
    conf, correct = p.max(1), p.argmax(1) == y
    ts = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    cov = np.array([(conf >= t).mean() for t in ts])
    acc = np.array([correct[conf >= t].mean() if (conf >= t).any() else np.nan for t in ts])
    return ts, cov, acc


def pick_threshold(p, y):
    return max(_val_threshold(p, y), C.THRESHOLD_FLOOR)   # val-calibrated value, never below the documented floor


def _val_threshold(p, y):
    ts, cov, acc = selective(p, y)
    n_cov = np.array([(p.max(1) >= t).sum() for t in ts])
    ok = np.nonzero((acc >= C.THRESHOLD_TARGET_ACC) & (n_cov >= 20))[0]
    return float(ts[ok[0]]) if len(ok) else float(ts[np.nanargmax(np.where(cov >= 0.5, acc, np.nan))])


def summarize(p, y, t=None):
    pred = p.argmax(1)
    res = {"n": int(len(y)), "accuracy": float((pred == y).mean()) if len(y) else None}
    if len(y):
        rep = classification_report(y, pred, labels=list(range(NC)), target_names=C.CLASSES,
                                    output_dict=True, zero_division=0)
        res["per_class"] = {c: {k: round(v, 4) for k, v in rep[c].items()} for c in C.CLASSES}
        present = sorted(set(y.tolist()))
        res["macro_f1_present_classes"] = float(np.mean([rep[C.CLASSES[i]]["f1-score"] for i in present]))
        res["confusion"] = confusion_matrix(y, pred, labels=list(range(NC))).tolist()
        res["false_button_alarms"] = int(((pred == CIDX["button_cell"]) & (y != CIDX["button_cell"])).sum())
        if t is not None:
            m = p.max(1) >= t
            res["at_threshold"] = {"threshold": t, "coverage": float(m.mean()),
                                   "accuracy_on_covered": float((pred[m] == y[m]).mean()) if m.any() else None,
                                   "errors_let_through": int(((pred != y) & m).sum())}
    return res


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.grid(False)
    ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
    for i in range(NC):
        for j in range(NC):
            if cm[i].sum() == 0:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED)
                continue
            ax.text(j, i, f"{cm[i, j]}\n{rown[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if rown[i, j] > 0.6 else INK)
    ax.set_xticks(range(NC), C.CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(NC), [f"{c} (n={cm[i].sum()})" for i, c in enumerate(C.CLASSES)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_coverage(pv, yv, pt, yt, t, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for p, y, ls, name in ((pv, yv, "-", "val"), (pt, yt, "--", "test")):
        ts, cov, acc = selective(p, y)
        ax.plot(ts, acc, ls, color=BLUE, lw=2, label=f"accuracy on accepted ({name})")
        ax.plot(ts, cov, ls, color=ORANGE, lw=2, label=f"coverage = share accepted ({name})")
    ax.axvline(t, color=INK2, lw=1)
    ax.text(t + 0.01, 0.02, f"recommended {t:.2f}", color=INK2, fontsize=9)
    ax.set_xlabel("confidence threshold (below it: 'uncertain')")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.08))
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training(variants, path):
    fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 3.8), squeeze=False)
    for ax, v in zip(axes[0], variants):
        logs = []
        for phase in ("head", "ft"):
            f = C.RUNS / v / f"log_{phase}.csv"
            if f.exists():
                logs += [dict(r, phase=phase) for r in csv.DictReader(open(f))]
        x = np.arange(1, len(logs) + 1)
        ax.plot(x, [float(r["loss"]) for r in logs], color=BLUE, lw=2, label="train loss (augmented)")
        ax.plot(x, [float(r["val_loss"]) for r in logs], color=ORANGE, lw=2, label="val loss")
        nh = sum(1 for r in logs if r["phase"] == "head")
        ax.axvline(nh + 0.5, color=MUTED, lw=1)
        ax.text(nh + 0.7, ax.get_ylim()[1] * 0.92, "fine-tune starts", color=MUTED, fontsize=8)
        ax.set_title(f"{v}", loc="left", fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def main():
    C.REPORTS.mkdir(exist_ok=True)
    rows = [r for r in csv.DictReader(open(C.WORK / "splits.csv", encoding="utf-8"))]
    variants = [v for v in ("baseline", "synthetic") if (C.RUNS / v / "export" / "export.json").exists()]
    if not variants:
        sys.exit("no exported variant found")
    size = json.load(open(C.RUNS / variants[0] / "done.json"))["img_size"]

    sets = {}
    for s in ("val", "test", "web_holdout"):
        rs = [r for r in rows if r["split"] == s]
        X = np.stack([load_image(C.ROOT / r["path"], size) for r in rs]) if rs else np.zeros((0, size, size, 3), np.uint8)
        sets[s] = (X, np.array([CIDX[r["cls"]] for r in rs], dtype=int), rs)
    Xd, yd = build_diagnostic(rows, size)
    sets["diagnostic"] = (Xd, yd, None)
    print("[eval] set sizes:", {k: len(v[1]) for k, v in sets.items()})

    results = {}
    for v in variants:
        for m in ("battery_float32", "battery_int8"):
            path = C.RUNS / v / "export" / f"{m}.tflite"
            probs = {s: predict(path, X) if len(X) else np.zeros((0, NC)) for s, (X, _, _) in sets.items()}
            t = pick_threshold(probs["val"], sets["val"][1])
            r = {s: summarize(probs[s], sets[s][1], t) for s in sets}
            ts, cov, acc = selective(probs["val"], sets["val"][1])
            r["threshold_table_val"] = {f"{x:.2f}": {"coverage": float(c), "accuracy": float(a)}
                                        for x, c, a in zip(ts, cov, acc) if round(x * 100) % 5 == 0}
            for s in ("test", "diagnostic"):
                ts, cov, acc = selective(probs[s], sets[s][1])
                r[f"threshold_table_{s}"] = {f"{x:.2f}": {"coverage": float(c), "accuracy": float(a)}
                                             for x, c, a in zip(ts, cov, acc) if round(x * 100) % 5 == 0}
            r["threshold_val_calibrated"] = _val_threshold(probs["val"], sets["val"][1])
            r["threshold"] = t
            by_src = {}
            for src in sorted({x["source"] for x in sets["test"][2]}):
                idx = np.array([x["source"] == src for x in sets["test"][2]])
                by_src[src] = summarize(probs["test"][idx], sets["test"][1][idx])["accuracy"]
            r["test_accuracy_by_source"] = by_src
            hb = sets["web_holdout"][1] == CIDX["button_cell"]
            r["web_holdout_button_recall"] = float((probs["web_holdout"].argmax(1)[hb] == CIDX["button_cell"]).mean()) if hb.any() else None
            r["web_holdout_predicted_as"] = dict(Counter(C.CLASSES[i] for i in probs["web_holdout"].argmax(1)))
            r["size_kb"] = round(path.stat().st_size / 1024, 1)
            results[f"{v}/{m}"] = r
            np.savez_compressed(C.REPORTS / f"probs_{v}_{m}.npz", **{s: probs[s] for s in probs})
            print(f"  {v}/{m}: val acc {pct(r['val']['accuracy'])}, test acc {pct(r['test']['accuracy'])}, "
                  f"web-holdout button recall {pct(r['web_holdout_button_recall'])}, diag acc {pct(r['diagnostic']['accuracy'])}, t={t}")

    # ---- choose variant (validation + web holdout only; test is never used for this) ----
    chosen = "baseline"
    synth_note = "synthetic variant not trained"
    if "synthetic" in variants:
        b, s = results["baseline/battery_float32"], results["synthetic/battery_float32"]
        gain = (s["web_holdout_button_recall"] or 0) - (b["web_holdout_button_recall"] or 0)
        drop = b["val"]["accuracy"] - s["val"]["accuracy"]
        rule = C.SYNTH_KEEP_RULE
        if C.SYNTHETIC_MODE == "on" or (gain >= rule["min_button_recall_gain"] and drop <= rule["max_val_acc_drop"]):
            chosen = "synthetic"
        synth_note = (f"web-holdout button recall {pct(b['web_holdout_button_recall'])} -> {pct(s['web_holdout_button_recall'])} "
                      f"(gain {100 * gain:+.1f} pts); val accuracy {pct(b['val']['accuracy'])} -> {pct(s['val']['accuracy'])} "
                      f"(drop {100 * drop:+.1f} pts) -> {'KEEP synthetic' if chosen == 'synthetic' else 'keep baseline'}")
    f32, i8 = results[f"{chosen}/battery_float32"], results[f"{chosen}/battery_int8"]
    int8_drop = f32["test"]["accuracy"] - i8["test"]["accuracy"]
    recommended = "battery_int8" if int8_drop <= C.INT8_MAX_ACC_DROP else "battery_float32"
    rec = results[f"{chosen}/{recommended}"]

    # ---- deliverables ----
    C.MODELS.mkdir(exist_ok=True)
    for m in ("battery_float32", "battery_int8"):
        shutil.copy(C.RUNS / chosen / "export" / f"{m}.tflite", C.MODELS / f"{m}.tflite")
    (C.MODELS / "labels.txt").write_text("\n".join(C.CLASSES) + "\n", encoding="utf-8")
    shutil.copy(C.ROOT / "infer_pi.py", C.MODELS / "infer_pi.py")
    info = {"created": datetime.now().isoformat(timespec="seconds"), "variant": chosen,
            "recommended_model": f"{recommended}.tflite", "img_size": size, "classes": C.CLASSES,
            "thresholds": {f"{m}.tflite": results[f"{chosen}/{m}"]["threshold"] for m in ("battery_float32", "battery_int8")},
            "preprocessing": "EXIF-rotate, RGB, pad to square with median border colour, bilinear resize to "
                             f"{size}x{size}; float32 model takes 0-255 float, int8 model takes 0-255 uint8",
            "button_cell_reliable": any(r["cls"] == "button_cell" and r["origin"] == "own" for r in rows)}
    json.dump(info, open(C.MODELS / "model_info.json", "w"), indent=2)

    # ---- plots ----
    plot_confusion(rec["test"]["confusion"], f"Test set - {chosen}/{recommended}", C.REPORTS / "confusion_matrix_test.png")
    plot_confusion(f32["test"]["confusion"], f"Test set - {chosen}/battery_float32", C.REPORTS / "confusion_matrix_test_float32.png")
    plot_confusion(rec["diagnostic"]["confusion"], "Shortcut diagnostic (cut-outs on tray) - NOT the test set",
                   C.REPORTS / "confusion_matrix_diagnostic.png")
    pv = np.load(C.REPORTS / f"probs_{chosen}_{recommended}.npz")
    plot_coverage(pv["val"], sets["val"][1], pv["test"], sets["test"][1], rec["threshold"],
                  f"Coverage vs accuracy - {chosen}/{recommended}", C.REPORTS / "threshold_tradeoff.png")
    plot_training(variants, C.REPORTS / "training_curves.png")

    split_stats = json.load(open(C.WORK / "split_stats.json"))
    summary = dict(chosen_variant=chosen, recommended_model=recommended, int8_test_acc_drop=int8_drop,
                   synthetic_experiment=synth_note, results=results, split_stats=split_stats,
                   train_meta={v: json.load(open(C.RUNS / v / "done.json")) for v in variants},
                   export_meta={v: json.load(open(C.RUNS / v / "export" / "export.json")) for v in variants})
    json.dump(summary, open(C.REPORTS / "metrics.json", "w"), indent=2)
    write_results_md(summary, info)
    print(f"[eval] shipped {chosen}/{recommended} (threshold {rec['threshold']}); int8 test-acc drop {100 * int8_drop:.2f} pts")
    print(f"[eval] synthetic experiment: {synth_note}")


def write_results_md(s, info):
    L = [f"# RESULTS (auto-generated {info['created']})", "",
         "Regenerated by every pipeline run. REPORT.md explains the decisions; this file holds the latest numbers.", ""]
    if not info["button_cell_reliable"]:
        L += ["> **button_cell metrics are UNRELIABLE.** There are no own photos of button cells yet, so button_cell is "
              "absent from val/test; its numbers below come only from held-out Wikimedia photos (web_holdout).", ""]
    L += [f"Shipped: **{s['chosen_variant']}/{s['recommended_model']}.tflite**; int8 vs float32 test-accuracy drop: "
          f"{100 * s['int8_test_acc_drop']:.2f} pts", f"Synthetic experiment: {s['synthetic_experiment']}", "",
          "## Image counts (used images per class / source / split)", "", "| class / source | train | val | test | web_holdout |",
          "|---|---|---|---|---|"]
    for k, v in s["split_stats"]["counts"].items():
        L.append(f"| {k.replace('|', ' / ')} | {v.get('train', 0)} | {v.get('val', 0)} | {v.get('test', 0)} | {v.get('web_holdout', 0)} |")
    L += ["", "## Model comparison", "",
          "| model | size KB | val acc | test acc | test macro-F1 (classes present) | web-holdout button recall (UNRELIABLE) | shortcut-diagnostic acc | threshold | test coverage @t | test acc @t |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in s["results"].items():
        at = r["test"].get("at_threshold", {})
        L.append(f"| {name} | {r['size_kb']} | {pct(r['val']['accuracy'])} | {pct(r['test']['accuracy'])} | "
                 f"{r['test']['macro_f1_present_classes']:.3f} | {pct(r['web_holdout_button_recall'])} | "
                 f"{pct(r['diagnostic']['accuracy'])} | {r['threshold']:.2f} | {pct(at.get('coverage'))} | {pct(at.get('accuracy_on_covered'))} |")
    rec = s["results"][f"{s['chosen_variant']}/{s['recommended_model']}"]
    L += ["", f"## Per-class test metrics ({s['chosen_variant']}/{s['recommended_model']})", "",
          "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for c, m in rec["test"]["per_class"].items():
        note = " (no test images - see web_holdout)" if m["support"] == 0 else ""
        L.append(f"| {c}{note} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {int(m['support'])} |")
    L += ["", f"Overall test accuracy: **{pct(rec['test']['accuracy'])}** on {rec['test']['n']} images. "
          f"False 'button_cell' alarms on test: {rec['test']['false_button_alarms']}.",
          f"Test accuracy by source: {json.dumps({k: round(v, 4) for k, v in rec['test_accuracy_by_source'].items()})}",
          "", f"Web-holdout (button cells, UNRELIABLE): recall {pct(rec['web_holdout_button_recall'])}, "
          f"predicted as {rec['web_holdout_predicted_as']}", "",
          f"Shortcut diagnostic (held-out cut-outs pasted on the tray; NOT the test set): accuracy {pct(rec['diagnostic']['accuracy'])}; "
          "per-class recall " + json.dumps({c: m["recall"] for c, m in rec["diagnostic"]["per_class"].items() if m["support"]}), "",
          "![confusion](confusion_matrix_test.png)", "", "## Threshold trade-off (validation; test dashed)", "",
          "| threshold | val coverage | val acc on accepted | test coverage | test acc on accepted |", "|---|---|---|---|---|"]
    for t in ("0.50", "0.60", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"):
        a, b = rec["threshold_table_val"].get(t), rec["threshold_table_test"].get(t)
        if a and b:
            L.append(f"| {t} | {pct(a['coverage'])} | {pct(a['accuracy'])} | {pct(b['coverage'])} | {pct(b['accuracy'])} |")
    L += ["", "Same trade-off on the shortcut diagnostic (domain-shifted images, NOT the test set):", "",
          "| threshold | diag coverage | diag acc on accepted |", "|---|---|---|"]
    for t in ("0.50", "0.60", "0.70", "0.80", "0.90"):
        a = rec["threshold_table_diagnostic"].get(t)
        if a:
            L.append(f"| {t} | {pct(a['coverage'])} | {pct(a['accuracy'])} |")
    L += ["", f"Validation-calibrated threshold (lowest t with >= {100 * C.THRESHOLD_TARGET_ACC:.0f}% selective accuracy): "
          f"{rec['threshold_val_calibrated']:.2f}; floor {C.THRESHOLD_FLOOR:.2f}; "
          f"**shipped threshold: {rec['threshold']:.2f}**", "", "![tradeoff](threshold_tradeoff.png)", "",
          "![training](training_curves.png)", ""]
    (C.REPORTS / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
