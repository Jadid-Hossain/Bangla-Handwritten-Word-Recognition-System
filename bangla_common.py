"""Shared toolbox used by train.py and app.py: the fixed paths, the labels,
loading the saved model, character-image normalisation, word segmentation
and the whole-word prediction logic."""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
from scipy import ndimage

# ---- static paths: they never depend on the folder you start Python from ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.keras"
LABELS_PATH = BASE_DIR / "labels.json"
MLFLOW_DIR = BASE_DIR / "artifacts" / "mlflow"
TRACKING_URI = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = MLFLOW_DIR / "artifacts"
TRAIN_EXPERIMENT = "bangla-ocr-training"
SERVE_EXPERIMENT = "bangla-ocr-serving"

IMG_SIZE = 32  # must match the value the model was trained with

# classes 84..93 of the model are the vowel signs (kars). They are not part of
# the 84 BanglaLekha-Isolated classes, so train.py generates them synthetically
# from a font; the app needs them to read words such as "মানুষ".
KAR_CLASSES = ["া", "ি", "ী", "ু", "ূ", "ৃ", "ে", "ৈ", "ো", "ৌ"]
NUM_CLASSES = 84 + len(KAR_CLASSES)


def load_labels() -> dict[int, str]:
    """Model class index -> character string, from the static labels.json."""
    with open(LABELS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    labels = {int(k): v for k, v in raw.items()}
    if set(labels) != set(range(NUM_CLASSES)):
        raise ValueError(f"{LABELS_PATH.name} must map exactly the {NUM_CLASSES} classes")
    return labels


def setup_device() -> str:
    """Use the GPU when TensorFlow can see one, otherwise the CPU."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return "GPU" if gpus else "CPU"


def load_model():
    import tensorflow as tf
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run 'python train.py' first.")
    return tf.keras.models.load_model(MODEL_PATH)


def get_experiment_id(mlflow, name: str) -> str:
    """Create the experiment once, with its artifacts under the static MLFLOW_ARTIFACTS folder."""
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return mlflow.create_experiment(name, artifact_location=(MLFLOW_ARTIFACTS / name).as_uri())


# --------------------------------------------------------------------------
# character image normalisation: everything the model sees goes through this
# --------------------------------------------------------------------------
def _ink_mask(a: np.ndarray, ink_mask: np.ndarray = None) -> np.ndarray:
    """Boolean ink mask. When none is given (raw dataset images, font renders),
    ink is the minority polarity of the picture — a character covers far less
    than half of its rectangle in whichever polarity the ink is drawn."""
    if ink_mask is not None:
        return ink_mask
    dark_frac = float((a < 128).mean())
    return (a < 128) if dark_frac <= 0.5 else (a > 127)


def normalise_char(img: np.ndarray, img_size: int = IMG_SIZE,
                   ink_mask: np.ndarray = None) -> np.ndarray:
    """One character image -> (img_size, img_size) uint8, WHITE ink on BLACK.

    Accepts a 2-D greyscale array of either polarity; pass ink_mask when the
    ink is already known (pieces cut from a canvas). The ink is cropped to its
    bounding box, a 10 % margin is added, the picture is padded to a square and
    resized with Lanczos — the same function for training images and canvas
    pieces, so the model always sees the same kind of picture.
    """
    a = np.asarray(img, dtype=np.uint8)
    ink = _ink_mask(a, ink_mask)
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return np.zeros((img_size, img_size), dtype=np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = a[y0:y1, x0:x1]
    ink_crop = ink[y0:y1, x0:x1]
    if float(np.median(crop[ink_crop])) < 128:
        crop = 255 - crop  # dark ink -> invert to white ink on black
    h, w = crop.shape
    m = max(round(0.10 * max(h, w)), 2)
    side = max(h, w) + 2 * m
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop
    out = Image.fromarray(canvas).resize((img_size, img_size), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


# --------------------------------------------------------------------------
# word segmentation: split the canvas drawing into character pieces
# --------------------------------------------------------------------------
def _boxes_from_mask(mask: np.ndarray) -> list[dict]:
    """Connected components of the ink mask -> bounding boxes with metadata."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        area = int((lab[sl] == i + 1).sum())
        if area < 12:  # speck of noise
            continue
        boxes.append({
            "y0": sl[0].start, "x0": sl[1].start,
            "y1": sl[0].stop, "x1": sl[1].stop,
            "area": area,
        })
    boxes.sort(key=lambda b: b["x0"])
    # drop specks fully inside another box (the counter/hole of ০, ৮, ৯ when a
    # thin stroke disconnects it): contained and much smaller than its host
    kept: list[dict] = []
    for b in boxes:
        host = None
        for m in boxes:
            if m is b or m["area"] <= b["area"]:
                continue
            contained = (m["x0"] <= b["x0"] and m["y0"] <= b["y0"]
                         and m["x1"] >= b["x1"] and m["y1"] >= b["y1"])
            if contained and b["area"] < 0.35 * m["area"]:
                host = m
                break
        if host is None:
            kept.append(b)
    return kept


def _group_cells(boxes: list[dict]) -> list[list[dict]]:
    """Group boxes into reading cells: boxes whose x-ranges overlap heavily
    belong to the same cell (a letter and a vowel sign drawn on it)."""
    cells: list[list[dict]] = []
    for b in boxes:
        placed = False
        for cell in cells:
            if any(
                min(b["x1"], m["x1"]) - max(b["x0"], m["x0"])
                > 0.35 * min(b["x1"] - b["x0"], m["x1"] - m["x0"])
                for m in cell
            ):
                cell.append(b)
                placed = True
                break
        if not placed:
            cells.append([b])
    cells.sort(key=lambda c: min(b["x0"] for b in c))
    for cell in cells:
        cell.sort(key=lambda b: b["x0"])
    return cells


def _split_wide(crop: np.ndarray, ink: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a wide glyph (e.g. ম and া drawn as one stroke) at the thinnest
    column near the middle, so the two halves can be classified separately.
    Returns (crop, mask) pieces."""
    cols = ink.sum(axis=0).astype(float)
    w = len(cols)
    lo, hi = int(w * 0.30), int(w * 0.70)
    if hi - lo < 4:
        return [(crop, ink)]
    smooth = np.convolve(cols, np.ones(3) / 3, mode="same")
    cut = lo + int(np.argmin(smooth[lo:hi]))
    if smooth[cut] > 0.25 * max(cols.max(), 1):
        return [(crop, ink)]  # no convincing split point
    out = []
    for sl in (slice(None, cut), slice(cut, None)):
        c, m = crop[:, sl], ink[:, sl]
        if c.size and m.sum() >= 12:
            out.append((c, m))
    return out or [(crop, ink)]


def classify_pieces(pieces: list[tuple[np.ndarray, np.ndarray]], model,
                    labels: dict[int, str]) -> list[dict]:
    """Classify normalised character pieces in one batch. Every piece is a
    (greyscale crop, ink mask) pair cut out of the canvas."""
    preds: list[dict] = []
    if not pieces:
        return preds
    batch = np.stack([normalise_char(c, ink_mask=m) for c, m in pieces]).astype("float32")
    batch = batch.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    probs = model(batch, training=False).numpy()
    for (c, m), row in zip(pieces, probs):
        idx = int(row.argmax())
        preds.append({
            "char": labels[idx],
            "conf": float(row[idx]),
            "probs": row,
            "image": normalise_char(c, ink_mask=m),
        })
    return preds


def recognize_word(rgba: np.ndarray, model, labels: dict[int, str]) -> dict:
    """Canvas RGBA (white stroke on black) -> predicted whole word.

    Steps: threshold -> connected components -> reading cells -> classify each
    piece (splitting very wide pieces once) -> join characters into a word,
    putting every vowel sign after the letter it is attached to (Unicode order).
    """
    arr = np.asarray(rgba).astype(np.uint8)
    gray = arr[:, :, :3].max(axis=2) if arr.ndim == 3 and arr.shape[2] >= 3 else arr
    mask = gray > 40
    boxes = _boxes_from_mask(mask)
    result = {
        "word": "",
        "pieces": [],   # per-piece predictions in reading order
        "cells": [],    # per-cell piece counts (for display)
        "n_components": len(boxes),
        "mean_conf": 0.0,
    }
    if not boxes:
        return result

    cells = _group_cells(boxes)
    main_height = float(np.median([b["y1"] - b["y0"] for b in boxes]))

    word_chars: list[str] = []
    for cell in cells:
        heights = [b["y1"] - b["y0"] for b in cell]
        cell_max_h = max(heights)
        mains = [b for b in cell if (b["y1"] - b["y0"]) >= 0.55 * cell_max_h]
        mains.sort(key=lambda b: b["x0"])
        main_ids = {id(b) for b in mains}

        # small pieces sitting on TOP of a letter with a big x-overlap (the
        # headline bar of থ, the crescent of ি, the dot of ং) belong to that
        # letter: they are merged into its picture before classification.
        attached: dict[int, list[dict]] = {id(m): [] for m in mains}
        rest: list[dict] = []
        for b in cell:
            if id(b) in main_ids:
                continue
            bw = b["x1"] - b["x0"]
            host = None
            for m in mains:
                ov = min(b["x1"], m["x1"]) - max(b["x0"], m["x0"])
                mh = m["y1"] - m["y0"]
                if ov > 0.55 * bw and b["y1"] <= m["y0"] + 0.55 * mh:
                    host = m
                    break
            if host is not None:
                attached[id(host)].append(b)
            else:
                rest.append(b)

        cell_results: list[dict] = []
        for m in mains:
            grp = [m] + attached[id(m)]
            x0 = min(b["x0"] for b in grp)
            y0 = min(b["y0"] for b in grp)
            x1 = max(b["x1"] for b in grp)
            y1 = max(b["y1"] for b in grp)
            crop = gray[y0:y1, x0:x1]
            mcrop = mask[y0:y1, x0:x1]
            w, h = x1 - x0, y1 - y0
            results = classify_pieces([(crop, mcrop)], model, labels)
            # a very wide piece is probably two letters drawn as one stroke
            # (e.g. ম and া touching): split it and keep the two-letter reading
            # when both halves are read confidently, else keep the whole piece
            if w > 1.15 * h and w > 1.10 * main_height:
                pieces = _split_wide(crop, mcrop)
                if len(pieces) > 1:
                    parts = classify_pieces(pieces, model, labels)
                    part_conf = min(r["conf"] for r in parts) if parts else 0.0
                    whole_conf = results[0]["conf"] if results else 0.0
                    if part_conf >= 0.50 and part_conf >= whole_conf - 0.20:
                        results = parts
            for r in results:
                r["bbox"] = (x0, y0, x1, y1)
            cell_results.extend(results)

        # remaining small pieces are vowel signs / dots read on their own
        for b in sorted(rest, key=lambda b: b["x0"]):
            crop = gray[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            mcrop = mask[b["y0"]:b["y1"], b["x0"]:b["x1"]]
            results = classify_pieces([(crop, mcrop)], model, labels)
            for r in results:
                r["bbox"] = (b["x0"], b["y0"], b["x1"], b["y1"])
            cell_results.extend(results)

        # Unicode order: every vowel sign comes after the letter it belongs to.
        # A kar drawn to the LEFT of its letter (ি ে …) is therefore deferred
        # until the letter itself has been emitted.
        line: list[str] = []
        pending: list[str] = []
        for r in cell_results:
            if r["char"] in KAR_CLASSES:
                pending.append(r["char"])
                continue
            line.append(r["char"])
            line.extend(pending)
            pending = []
        line.extend(pending)
        # small repair for vowel signs drawn in two pieces: ে+া -> ো, ে+ু -> ৌ
        text = "".join(line).replace("ো", "ো").replace("েু", "ৌ")
        result["cells"].append(len(cell_results))
        word_chars.append(text)
        result["pieces"].extend(cell_results)

    result["word"] = "".join(word_chars)
    result["mean_conf"] = float(np.mean([r["conf"] for r in result["pieces"]]))
    return result
