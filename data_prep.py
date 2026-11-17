"""Data preparation for train.py, kept free of TensorFlow so the worker
processes of the parallel build start quickly: reading the BanglaLekha-Isolated
images, synthesising the vowel-sign (kar) classes from the bundled font, and
writing the cache file everything trains from."""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bangla_common import IMG_SIZE, KAR_CLASSES, normalise_char

N_REAL_CLASSES = 84
KARS_PER_CLASS = 2200
WORKERS = max(1, (os.cpu_count() or 2) - 1)

FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSansBengali-Regular.ttf"
IMAGES_DIR = Path(__file__).resolve().parent / "data" / "BanglaLekha-Isolated" / "Images"


# --------------------------------------------------------------------------
# worker functions (must stay module-level for Windows spawn)
# --------------------------------------------------------------------------
def _load_folder(task: tuple[int, str]):
    """Read one dataset class folder -> normalised images."""
    label_id, folder = task
    from PIL import Image
    files = sorted(Path(folder).glob("*.png"))
    out = np.zeros((len(files), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for k, f in enumerate(files):
        img = Image.open(f).convert("L")
        out[k] = normalise_char(np.asarray(img, dtype=np.uint8))
    return label_id, out


def _kar_chunk(task: tuple[int, int, int]):
    """Render + augment a chunk of one kar class. The per-image rng seed makes
    the synthesis deterministic no matter which worker draws which image."""
    class_id, start_i, count = task
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    kar = KAR_CLASSES[class_id]
    out = np.zeros((count, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    fonts: dict[int, ImageFont.FreeTypeFont] = {}
    for j in range(count):
        rng = np.random.default_rng(100_000 + 97 * class_id + start_i + j)
        size = int(rng.integers(34, 92))
        if size not in fonts:
            fonts[size] = ImageFont.truetype(str(FONT_PATH), size)
        if kar in ("ো", "ৌ"):  # two-part signs: place the parts side by side by hand
            arr = _render_two_part(kar, fonts[size], rng)
        else:
            img = Image.new("L", (200, 200), 255)
            ImageDraw.Draw(img).text((50, 50), kar, font=fonts[size], fill=0)
            arr = np.asarray(img, dtype=np.uint8)
        # random rotation / shear / scale to imitate handwriting variety
        angle = np.deg2rad(rng.uniform(-8, 8))
        shear = rng.uniform(-0.12, 0.12)
        scale = rng.uniform(0.85, 1.15)
        matrix = (np.cos(angle) / scale, -np.sin(angle) / scale + shear, rng.uniform(-6, 6),
                  np.sin(angle) / scale + shear, np.cos(angle) / scale, rng.uniform(-6, 6))
        arr = np.asarray(Image.fromarray(arr).transform(
            (200, 200), Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=255),
            dtype=np.uint8)
        ink = arr < 128
        r = rng.random()
        if r < 0.4:
            arr = (255 - 255 * ndimage.binary_dilation(ink, iterations=1)).astype(np.uint8)
        elif r < 0.7:
            arr = (255 - 255 * ndimage.binary_erosion(ink, iterations=1)).astype(np.uint8)
        if rng.random() < 0.5:
            arr = ndimage.gaussian_filter(arr, rng.uniform(0.0, 0.8)).astype(np.uint8)
        out[j] = normalise_char(arr)
    return class_id, start_i, out


def _render_two_part(kar: str, font, rng) -> np.ndarray:
    """ো and ৌ decompose into a left part and a right part; fonts render them
    with a dotted circle when drawn alone, so the parts are placed by hand."""
    from PIL import Image, ImageDraw

    def part(ch: str) -> np.ndarray:
        img = Image.new("L", (200, 200), 255)
        ImageDraw.Draw(img).text((50, 50), ch, font=font, fill=0)
        a = np.asarray(img, dtype=np.uint8)
        ys, xs = np.where(a < 128)
        return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    right_char = "া" if kar == "ো" else "ু"
    left = part("ে")
    right = part(right_char)
    # the right part is a bit smaller and sits lower, like on a drawing
    rh = max(2, int(right.shape[0] * rng.uniform(0.75, 0.9)))
    rw = max(2, int(right.shape[1] * rh / right.shape[0]))
    right = np.asarray(Image.fromarray(right).resize((rw, rh), Image.LANCZOS))
    gap = int(rng.integers(-4, 8))  # may slightly touch or slightly overlap
    top_pad = int(rng.integers(0, 8))
    h = max(left.shape[0], rh) + top_pad + 4
    board = np.full((h, left.shape[1] + gap + rw + 8), 255, dtype=np.uint8)
    ly = int(rng.integers(0, max(1, h - left.shape[0] + 1)))
    board[ly:ly + left.shape[0], :left.shape[1]] = np.minimum(
        board[ly:ly + left.shape[0], :left.shape[1]], left)
    ry = h - rh  # bottom-aligned, never out of range
    rx = left.shape[1] + gap
    board[ry:ry + rh, rx:rx + rw] = np.minimum(board[ry:ry + rh, rx:rx + rw], right)
    return board


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def ensure_dataset(zip_path: Path, data_dir: Path, dataset_url: str) -> None:
    """Extract the dataset zip (or download it) if the Images folder is missing."""
    if IMAGES_DIR.exists():
        return
    if not zip_path.exists():
        import urllib.request
        print(f"==> Downloading BanglaLekha-Isolated from Mendeley (~188 MB)")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(dataset_url, zip_path)
    print(f"==> Extracting {zip_path.name}")
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)


def load_real_images() -> tuple[np.ndarray, np.ndarray]:
    """All 84 dataset folders -> (images (N,s,s) uint8, labels (N,) int16)."""
    folders = sorted([p for p in IMAGES_DIR.iterdir() if p.is_dir()], key=lambda p: int(p.name))
    if len(folders) != N_REAL_CLASSES:
        raise RuntimeError(f"Expected {N_REAL_CLASSES} class folders in {IMAGES_DIR}, found {len(folders)}")
    tasks = [(i, str(f)) for i, f in enumerate(folders)]
    by_label: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for label_id, arr in ex.map(_load_folder, tasks):
            by_label[label_id] = arr
            print(f"\r  folder {label_id + 1:2d}/84 read ({len(arr)} images)", end="", flush=True)
    print()
    counts = [len(by_label[i]) for i in range(N_REAL_CLASSES)]
    images = np.concatenate([by_label[i] for i in range(N_REAL_CLASSES)])
    labels = np.repeat(np.arange(N_REAL_CLASSES, dtype=np.int16), counts)
    return images, labels


def augment_real(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random rotation / shear / scale / stroke thickness for a white-ink-on-black
    character, making the model robust to how a user draws on the canvas."""
    from PIL import Image
    from scipy import ndimage
    h, w = img.shape
    angle = np.deg2rad(rng.uniform(-8, 8))
    shear = rng.uniform(-0.10, 0.10)
    scale = rng.uniform(0.85, 1.15)
    dx, dy = rng.uniform(-2, 2), rng.uniform(-2, 2)
    a = np.deg2rad(angle)
    matrix = (np.cos(a) / scale, -np.sin(a) / scale + shear, dx,
              np.sin(a) / scale + shear, np.cos(a) / scale, dy)
    out = Image.fromarray(img).transform((w, h), Image.AFFINE, matrix,
                                         resample=Image.BILINEAR, fillcolor=0)
    ink = np.asarray(out) > 100
    r = rng.random()
    if r < 0.35:
        out = (255 * ndimage.binary_dilation(ink, iterations=1)).astype(np.uint8)
    elif r < 0.65:
        out = (255 * ndimage.binary_erosion(ink, iterations=1)).astype(np.uint8)
    else:
        out = (255 * ink).astype(np.uint8)
    return out


def load_augmented_real(x_real: np.ndarray, seed: int = 7) -> np.ndarray:
    """One augmented copy of every real image (same labels). Accepts (N,s,s) or (N,s,s,1)."""
    rng = np.random.default_rng(seed)
    if x_real.ndim == 4:
        x_real = x_real.reshape(-1, x_real.shape[1], x_real.shape[2])
    out = np.empty((len(x_real), x_real.shape[1], x_real.shape[2]), dtype=np.uint8)
    for i in range(len(x_real)):
        out[i] = normalise_char(augment_real(x_real[i], rng))
        if i % 30000 == 0:
            print(f"  augmented {i}/{len(x_real)}", end="\r", flush=True)
    print()
    return out


def load_kar_images() -> tuple[np.ndarray, np.ndarray]:
    """Synthesise the 10 vowel-sign classes from the bundled Noto Sans Bengali font."""
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"{FONT_PATH} is needed to synthesise the vowel-sign classes")
    chunk = KARS_PER_CLASS // 4
    tasks = []
    for class_id in range(len(KAR_CLASSES)):
        for c in range(4):
            start = c * chunk
            tasks.append((class_id, start, chunk if c < 3 else KARS_PER_CLASS - start))
    by_class: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(len(KAR_CLASSES))}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        done = 0
        for class_id, start_i, arr in ex.map(_kar_chunk, tasks):
            by_class[class_id].append((start_i, arr))
            done += 1
            print(f"\r  kar chunks done: {done}/{len(tasks)}", end="", flush=True)
    print()
    images, labels = [], []
    for class_id in range(len(KAR_CLASSES)):
        pieces = sorted(by_class[class_id], key=lambda t: t[0])
        images.append(np.concatenate([p[1] for p in pieces]))
        labels.append(np.full(KARS_PER_CLASS, N_REAL_CLASSES + class_id, dtype=np.int16))
        print(f"  kar '{KAR_CLASSES[class_id]}': {KARS_PER_CLASS} images")
    return np.concatenate(images), np.concatenate(labels)


def build_cache(cache_path: Path) -> None:
    """Read the dataset + synthesise the kar classes once, save to a single npz."""
    data_dir = cache_path.parent
    data_dir.mkdir(exist_ok=True)
    print("==> Reading the BanglaLekha-Isolated dataset")
    x_real, y_real = load_real_images()
    print(f"  real images: {len(x_real)}")
    print("==> Synthesising the vowel-sign classes")
    x_kar, y_kar = load_kar_images()
    x = np.concatenate([x_real, x_kar])
    y = np.concatenate([y_real, y_kar])
    order = np.random.default_rng(42).permutation(len(x))
    x, y = x[order], y[order]
    np.savez(cache_path, images=x, labels=y)
    print(f"==> Saved cache: {cache_path.name} ({cache_path.stat().st_size / 1e6:.0f} MB), {len(x)} images")
