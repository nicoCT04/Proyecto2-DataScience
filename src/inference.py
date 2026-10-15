"""Inferencia sobre video nuevo con los cuatro modelos entrenados.

Lo usa la aplicación: el usuario solo entrega un .mp4 y aquí ocurre, sin que lo vea,
el mismo preprocesamiento que en el entrenamiento (ventana de 1.62 s, 16 cuadros,
recorte del marcador, redimensionado) y la misma extracción de variables (OpenCV,
YOLOv8 y ResNet18). Después cada modelo devuelve sus probabilidades por clase.

  - Si el archivo conserva el nombre de un subclip del dataset
    (<partido>_<tiempo>_<ms evento>_<ms siguiente>.mp4) se clasifica la ventana del evento.
  - Si es cualquier otro video se recorre con una ventana deslizante y se obtiene una
    línea de tiempo de probabilidades.

Uso desde la raíz:
  python src/inference.py app/samples/<video>.mp4             # ventana del evento o recorrido
  python src/inference.py <video>.mp4 --t 12.5               # un momento concreto
"""
from __future__ import annotations

import os

# PyTorch y XGBoost traen cada uno su runtime de OpenMP; en macOS, con varios hilos,
# el proceso termina en segfault. Debe fijarse antes de importar cualquiera de los dos.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import modeling  # noqa: E402
import preprocessing as pp  # noqa: E402
import train_temporal  # noqa: E402
from features import window_features  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
MODELS = ["logreg", "rf", "xgb", "gru"]
NAMES = {"logreg": "Regresión logística", "rf": "Random Forest",
         "xgb": "XGBoost", "gru": "GRU (ResNet18)"}
LABELS_ES = {"background": "Fondo", "challenge": "Disputa", "play": "Pase", "throwin": "Saque de banda"}
FEATURE_COLS = modeling.feature_columns(pd.read_csv(ROOT / modeling.FEATURES_CSV, nrows=0))


def ensure_pretrained():
    """Descarga YOLOv8n y ResNet18 a models/pretrained/ la primera vez (no se versionan)."""
    features.PRETRAINED.mkdir(parents=True, exist_ok=True)
    if not features.YOLO_WEIGHTS.exists():
        torch.hub.download_url_to_file(YOLO_URL, str(features.YOLO_WEIGHTS), progress=False)
    if not features.RESNET_WEIGHTS.exists():
        torch.hub.download_url_to_file(features.RESNET_URL, str(features.RESNET_WEIGHTS), progress=False)


class Predictor:
    """Los cuatro modelos cargados una sola vez."""

    def __init__(self):
        ensure_pretrained()
        self.models = {k: joblib.load(ROOT / f"models/{k}.joblib") for k in ["logreg", "rf", "xgb"]}
        self.gru, _ = train_temporal.load_model()

    def predict(self, feats: pd.DataFrame, emb: np.ndarray, models=MODELS) -> dict[str, np.ndarray]:
        """feats: (n, variables) · emb: (n, 16, 512) -> {modelo: (n, 4) probabilidades}."""
        X = feats[FEATURE_COLS].to_numpy(float)
        out = {k: self.models[k].predict_proba(X) for k in models if k != "gru"}
        if "gru" in models:
            with torch.inference_mode():
                out["gru"] = torch.softmax(self.gru(torch.from_numpy(emb.astype(np.float32))), 1).numpy()
        return out


def subclip_event_time(path: str | Path, info: pp.VideoInfo) -> float | None:
    """Segundo del evento si el archivo tiene nombre de subclip del dataset; si no, None."""
    try:
        meta = pp.parse_subclip_name(Path(path).name)
    except (IndexError, ValueError):
        return None
    if meta["next_s"] <= meta["event_s"]:
        return None
    return pp.subclip_event_time(info, meta["event_s"], meta["next_s"])


def analyze_window(path: str | Path, t_s: float, info: pp.VideoInfo | None = None):
    """Preprocesa la ventana que termina en t_s y extrae sus variables."""
    t0 = time.perf_counter()
    win = pp.preprocess_window(path, t_s, info)
    feats, emb = window_features(win)
    feats.update({"t_s": t_s, "field_view": win.is_field_view,
                  "extract_s": time.perf_counter() - t0})
    return feats, emb


def to_frame(probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Probabilidades de una ventana como tabla modelo x clase, más el promedio de todos."""
    tab = pd.DataFrame({k: v[0] for k, v in probs.items()}, index=modeling.LABELS).T
    if len(tab) > 1:
        tab.loc["todos"] = tab.mean()
    return tab


def classify_moment(path, predictor: Predictor, t_s: float | None = None, models=MODELS):
    """Clasifica una ventana. Sin t_s usa el evento del subclip o el centro del video."""
    info = pp.video_info(path)
    if t_s is None:
        t_s = subclip_event_time(path, info)
    if t_s is None:
        t_s = info.duration_s / 2
    feats, emb = analyze_window(path, t_s, info)
    probs = predictor.predict(pd.DataFrame([feats]), emb[None], models)
    return to_frame(probs), feats


def scan_video(path, predictor: Predictor, models=MODELS, stride_s: float = 0.5, progress=None):
    """Recorre el video completo; una fila por ventana con la probabilidad de cada modelo."""
    info = pp.video_info(path)
    times = pp.sliding_times(info, stride_s)
    rows, embs = [], []
    for i, t in enumerate(times, 1):
        feats, emb = analyze_window(path, float(t), info)
        rows.append(feats)
        embs.append(emb)
        if progress:
            progress(i, len(times))
    feats = pd.DataFrame(rows)
    probs = predictor.predict(feats, np.stack(embs), models)
    out = feats[["t_s", "field_view", "extract_s"]].copy()
    for k, p in probs.items():
        for j, lab in enumerate(modeling.LABELS):
            out[f"{k}_{lab}"] = p[:, j]
        out[f"{k}_pred"] = modeling.decode(p.argmax(1))
    if len(probs) > 1:
        mean = np.mean(list(probs.values()), axis=0)
        for j, lab in enumerate(modeling.LABELS):
            out[f"todos_{lab}"] = mean[:, j]
        out["todos_pred"] = modeling.decode(mean.argmax(1))
    return out, feats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=None, help="segundo a clasificar")
    ap.add_argument("--scan", action="store_true", help="recorrer todo el video")
    args = ap.parse_args()
    predictor = Predictor()
    info = pp.video_info(args.video)
    if args.scan or (args.t is None and subclip_event_time(args.video, info) is None):
        out, _ = scan_video(args.video, predictor)
        cols = ["t_s", "field_view"] + [f"{k}_pred" for k in MODELS] + ["todos_pred"]
        print(out[cols].to_string(index=False))
        return
    tab, feats = classify_moment(args.video, predictor, args.t)
    print(f"Ventana que termina en t = {feats['t_s']:.2f} s · vista de cancha: {feats['field_view']} "
          f"· extracción {feats['extract_s']:.1f} s")
    print(tab.round(3).rename(columns=LABELS_ES).to_string())
    print("Predicción:", {k: LABELS_ES[tab.loc[k].idxmax()] for k in tab.index})


if __name__ == "__main__":
    main()
