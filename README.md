# Bangla Handwritten Word Recognition System


Assignment 04 — Bangla Handwritten Word Recognition System

**Name:** MD. JADID HOSSAIN SANIM
**Student ID:** 1000129287 


---

## 1. Approach

The system is a classic "segment → classify → join" pipeline built around the
[BanglaLekha-Isolated](https://data.mendeley.com/datasets/hf6sf8zrkc/2) dataset
(166,105 handwritten character images, 84 classes).

**Training (`train.py`)**

* Every image of the 84 dataset classes is cropped to its ink, padded to a
  square with a 10% margin and resized to 32×32 (white ink on black) — the same
  normalisation for training and for canvas pieces, so the model always sees
  the same kind of picture.
* The 84 classes alone cannot spell words like মানুষ, because the vowel signs
  (া ি ু ে …) are not part of the dataset. Ten extra **vowel-sign classes** are
  synthesised from the Noto Sans Bengali font (OFL license, bundled in
  `assets/fonts/`) with random rotation, shear, scale and stroke-thickness
  changes — **94 classes in total**.
* A small CNN (≈550k parameters, 32×32 input) is trained with Adam and
  cross-entropy. Pixels are scaled inside the model with a `Rescaling(1/255)`
  layer, so the saved model takes raw 0–255 pictures and `app.py` cannot get
  the scaling wrong.
* An `--augment 1` flag adds one augmented copy of every real training image
  (rotation, shear, scale, thickness), which makes the model much more robust
  to the way a person actually draws on a canvas.
* Every run logs its parameters, per-epoch metrics, the final accuracy and the
  model file to MLflow (`artifacts/mlflow/mlflow.db`, SQLite). The **best
  epoch** (by test accuracy) is checkpointed, not the last one.

**Prediction (`app.py` + `bangla_common.py`)**

1. Threshold the canvas drawing and find connected components.
2. Group components into **reading cells** by horizontal overlap; a small piece
   sitting on top of a letter (the headline bar of থ, the crescent of ি) is
   merged into that letter's picture before classification.
3. A piece that is much wider than tall (ম and া drawn as one stroke) is split
   at its thinnest column, and the two-letter reading is kept when both halves
   are classified confidently.
4. Each piece is classified by the CNN; vowel signs are placed **after** the
   letter they belong to (Unicode order: ম + া → মা), and ে+া / ে+ু pairs drawn
   as two pieces are repaired to ো / ৌ.
5. The characters are joined and shown as one word.



---

## 2. MLflow runs

Six training runs are recorded in `artifacts/mlflow/mlflow.db`
(experiment `bangla-ocr-training`); every run stores its parameters,
per-epoch metrics, confusion matrix and the trained model with it.
The first three runs were trained on CPU (WSL 2 Ubuntu, Python 3.10.12 /
TensorFlow 2.20 ; the final three were retrained with
the same code on a **Kaggle GPU**, with the full dataset and more epochs:

| Run | Arch | Epochs | Data | Device | Best test accuracy |
|---|---|---|---|---|---|
| `baseline` | base CNN |  **10** | 169k | WSL CPU → Kaggle GPU | **0.9239** |
| `deeper-bn` | BatchNorm + GAP |  **12** | 169k | WSL CPU → Kaggle GPU | **0.9285** |
| `baseline-aug` | base CNN |   **12** | 320k (augmented) | WSL CPU → Kaggle GPU | **0.9257** |

The shipped `models/model.keras` is the Kaggle **baseline-aug** checkpoint
(base CNN trained with augmentation): it trades a little clean-test accuracy
for much better behaviour on free-hand canvas drawings, which is what the app
actually receives. Every `model.keras` in the store was normalised with
`_tools/fix_keras_config.py` so the pinned TensorFlow 2.20 (Docker) loads the
files saved by Kaggle's newer Keras. The app's **MLflow tab** lists all runs;
the standalone UI is:

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000
```

Screenshot: `screenshots/mlflow_experiment.png`.

---

## 3. How to run

### Quick way — Docker (Linux container)

```bash
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
# open http://localhost:8501  →  draw a word  →  Predict word
```

or with Compose: `docker compose up -d` (same thing, background).

The image already contains the trained model and the MLflow store, so it works
right after the build. `requirements-docker.txt` uses `tensorflow-cpu` and no
Jupyter/matplotlib, so the image stays small. The container listens on
`0.0.0.0:8501` and passes a `/health` healthcheck.

### Train / run on the host (Linux, or Windows with WSL 2)

```bash
bash setup.sh                 # creates venv/, installs requirements.txt
source venv/bin/activate

# the dataset is downloaded & extracted automatically on first run
python train.py --prepare-only                                            # data cache
python train.py --run-name baseline --arch base --epochs 10               # run 1
python train.py --run-name baseline-aug --augment 1 --epochs 12           # run 2
python train.py --run-name deeper-bn --arch deeper --epochs 12 --lr 0.0015

streamlit run app.py           # http://localhost:8501
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000
```

`python train.py` alone trains with the default settings; every run is a new
row in MLflow. The `models/model.keras` of the best run is what the app loads.

**Drawing tips:** write one word, big; leave a small gap between letters; draw
a vowel sign close to its letter.

---

## 4. Project structure

```
bangla-ocr-assignment/
├── train.py               # trains the model, records everything in MLflow
├── app.py                 # Streamlit: draw a word → predicted word + MLflow tab
├── bangla_common.py       # shared paths, normalisation, segmentation, prediction
├── data_prep.py           # dataset reading + synthetic vowel-sign classes
├── labels.json            # 94 model classes → Unicode strings
├── requirements.txt       # training/app environment (pinned)
├── requirements-docker.txt# serving environment (tensorflow-cpu)
├── setup.sh               # one-time Linux/WSL setup (venv + requirements)
├── Dockerfile             # python:3.10.22-slim, non-root user, healthcheck
├── docker-compose.yml     # one command for the whole app
├── .dockerignore / .gitignore
├── assets/fonts/          # Noto Sans Bengali (OFL) for the kar classes
├── models/model.keras     # trained model (best checkpoint)
├── artifacts/mlflow/      # MLflow store: mlflow.db + run artifacts
└── screenshots/           # streamlit_app.png, mlflow_experiment.png
```

The dataset itself is **not** committed: `train.py` downloads
BanglaLekha-Isolated from Mendeley into `data/` on first use.

---

## 5. Limitations

* **Segmentation is heuristic.** Letters that touch each other are read as one
  piece; the split-at-thinnest-column rescue works for common cases (ম + া)
  but not for every handwriting style. Drawing with small gaps gives the best
  results.
* **No conjunct awareness while segmenting.** Compound characters (ক্ষ, স্ক…)
  are in the 94 classes, but a conjunct drawn very wide may be split into two
  wrong pieces.
* **Multi-part vowel signs** (ো, ৌ) are reconstructed from their pieces (ে+া →
  ো, ে+ু → ৌ); drawing them as one piece also works. Rare signs (ৃ, ৈ) have
  little synthetic training data and are the weakest classes.
* **Diacritic dots** (ং, ঃ, the dots of ড়/ঢ়) must be drawn attached; a
  detached dot is read as a separate character.
* **Word level ≈ character level^n:** at ~92-93% per character, a five-letter
  word is exactly right roughly two times out of three; short words are much
  more reliable. No language model is applied after recognition.
* Inference is instant on CPU, but accuracy is below the 95%+ that large
  published models reach on the 84-class subset.

---

Prepared for Assignment 04 · Skills for Industry Competitiveness and Innovation
Program (SICIP) · BRAC University.
