"""Streamlit app: draw one Bangla word, get the predicted word, and watch MLflow.

The Predict tab segments the drawing into character pieces, classifies each with
the trained CNN and joins the pieces into the whole word (e.g. "মানুষ").
The MLflow tab lists the training runs and every prediction the app has made.
"""
import os
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow import MlflowClient
from streamlit_drawable_canvas import st_canvas

from bangla_common import (
    BASE_DIR, KAR_CLASSES, LABELS_PATH, MLFLOW_ARTIFACTS, MLFLOW_DIR, MODEL_PATH,
    NUM_CLASSES, SERVE_EXPERIMENT, TRACKING_URI, TRAIN_EXPERIMENT, get_experiment_id,
    load_labels, load_model, recognize_word, setup_device,
)

st.set_page_config(page_title="Bangla Handwritten Word Recognition System", page_icon=None, layout="wide")

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(TRACKING_URI)
device = setup_device()


@st.cache_resource(show_spinner="Loading the model...")
def get_resources():
    return load_model(), load_labels()


@st.cache_resource
def start_mlflow_ui():
    """Optional: the full MLflow UI next to the app, on the same static store (START_MLFLOW_UI=1)."""
    host = os.environ.get("MLFLOW_UI_HOST", "127.0.0.1")
    port = os.environ.get("MLFLOW_UI_PORT", "5000")
    return subprocess.Popen([sys.executable, "-m", "mlflow", "ui", "--host", host, "--port", port,
                             "--backend-store-uri", TRACKING_URI,
                             "--default-artifact-root", MLFLOW_ARTIFACTS.as_uri()])


if os.environ.get("START_MLFLOW_UI") == "1":
    start_mlflow_ui()

if not MODEL_PATH.exists():
    st.error(f"No model at {MODEL_PATH}. Run `python train.py` first.")
    st.stop()
model, labels = get_resources()


def log_prediction(word: str, n_pieces: int, mean_conf: float) -> None:
    try:
        mlflow.set_experiment(experiment_id=get_experiment_id(mlflow, SERVE_EXPERIMENT))
        with mlflow.start_run(run_name=f"predict-{word}"):
            mlflow.log_params({"predicted_word": word, "n_pieces": n_pieces,
                               "source": "canvas", "device": device})
            mlflow.log_metric("mean_confidence", mean_conf)
    except Exception as exc:  # logging must never break the app itself
        st.caption(f"(MLflow logging skipped: {exc})")


def piece_montage(pieces: list[dict]) -> np.ndarray:
    """Side-by-side preview of the classified pieces (for the results panel)."""
    imgs = [p["image"] for p in pieces]
    pad = 2
    w = sum(i.shape[1] for i in imgs) + pad * (len(imgs) + 1)
    h = max(i.shape[0] for i in imgs) + 2 * pad
    board = np.zeros((h, w), dtype=np.uint8)
    x = pad
    for img in imgs:
        board[pad:pad + img.shape[0], x:x + img.shape[1]] = img
        x += img.shape[1] + pad
    return board


st.title("Bangla Handwritten Word Recognition System")
st.caption(
    f"Model: `{MODEL_PATH.relative_to(BASE_DIR)}` | {NUM_CLASSES} classes "
    f"(84 dataset + {len(KAR_CLASSES)} vowel signs) | device: `{device}` | "
    f"MLflow store: `{MLFLOW_DIR.relative_to(BASE_DIR)}`"
)

tab_predict, tab_mlflow = st.tabs(["Predict", "MLflow"])

with tab_predict:
    col_in, col_out = st.columns([3, 2])
    with col_in:
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)", stroke_width=12, stroke_color="#FFFFFF",
            background_color="#000000", height=300, width=720,
            drawing_mode="freedraw", key="canvas",
        )
        go = st.button("Predict word", type="primary", use_container_width=True)
    with col_out:
        if go and canvas.image_data is not None:
            with st.spinner("Reading the word..."):
                result = recognize_word(canvas.image_data, model, labels)
            if result["n_components"] == 0:
                st.warning("The canvas is empty. Draw a Bangla word first.")
            elif not result["word"]:
                st.warning("Could not read the drawing. Try thicker, more separated letters.")
            else:
                st.markdown(
                    f"<div style='padding:6px 0 2px 0;font-size:13px;color:#888'>Predicted word</div>"
                    f"<div style='font-size:52px;line-height:1.15'>{result['word']}</div>",
                    unsafe_allow_html=True,
                )
                st.metric("Mean confidence", f"{result['mean_conf'] * 100:.1f}%")
                st.caption(
                    f"{result['n_components']} stroke group(s) → "
                    f"{len(result['pieces'])} character piece(s)"
                )
                st.image(piece_montage(result["pieces"]), width=320, clamp=True)
                per_char = pd.DataFrame({
                    "piece": [p["char"] for p in result["pieces"]],
                    "confidence": [p["conf"] for p in result["pieces"]],
                })
                st.bar_chart(per_char.set_index("piece"), height=220)
                log_prediction(result["word"], len(result["pieces"]), result["mean_conf"])
        else:
            st.info("Draw a Bangla word on the left, then press **Predict word**.")

with tab_mlflow:
    client = MlflowClient()
    st.subheader("Training runs")
    train_exp = mlflow.get_experiment_by_name(TRAIN_EXPERIMENT)
    if train_exp is None:
        st.info("No training runs yet. Run `python train.py` (twice for two runs).")
    else:
        runs = mlflow.search_runs(experiment_ids=[train_exp.experiment_id],
                                  order_by=["start_time DESC"])
        cols = ["run_id", "tags.mlflow.runName", "params.arch", "params.epochs", "params.lr",
                "params.img_size", "params.device", "metrics.final_test_accuracy",
                "metrics.final_top3_accuracy", "metrics.train_seconds"]
        st.dataframe(runs[[c for c in cols if c in runs.columns]], use_container_width=True,
                     hide_index=True)
        if len(runs):
            latest = runs.iloc[0]["run_id"]
            hist = pd.DataFrame(
                {m.step: m.value for m in client.get_metric_history(latest, "test_accuracy")},
                index=["test_accuracy"],
            ).T
            hist.index.name = "epoch"
            st.caption("Test accuracy per epoch, latest run")
            st.line_chart(hist, height=240)

    st.subheader("Predictions logged by this app")
    serve_exp = mlflow.get_experiment_by_name(SERVE_EXPERIMENT)
    if serve_exp is None:
        st.info("No predictions yet. Draw a word on the Predict tab.")
    else:
        preds = mlflow.search_runs(experiment_ids=[serve_exp.experiment_id],
                                   order_by=["start_time DESC"], max_results=20)
        cols = ["start_time", "params.predicted_word", "params.n_pieces",
                "metrics.mean_confidence"]
        st.dataframe(preds[[c for c in cols if c in preds.columns]], use_container_width=True,
                     hide_index=True)
    st.caption(
        "Full MLflow UI: `mlflow ui --backend-store-uri " + TRACKING_URI + "` "
        "or start the app with `START_MLFLOW_UI=1` (http://localhost:5000)."
    )
