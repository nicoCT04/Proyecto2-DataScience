"""Extracción de features de una ventana preprocesada.

Tres familias de variables:
  1. Visuales y de movimiento (OpenCV): color, bordes, diferencia entre cuadros y
     flujo óptico de Farnebäck separando el movimiento de cámara del de jugadores.
  2. Jugadores y balón (YOLOv8n, clases COCO person y sports ball).
  3. Embeddings de ResNet18 preentrenada en ImageNet (512 por cuadro), que alimentan
     al modelo temporal como secuencia y a los modelos tabulares como resumen.
"""
from __future__ import annotations

import warnings

from pathlib import Path

import cv2
import numpy as np

from preprocessing import GREEN_HI, GREEN_LO, YOLO_WIDTH, Window, green_ratio

warnings.filterwarnings("ignore", category=UserWarning)

PERSON, BALL = 0, 32  # ids COCO
N_YOLO_FRAMES = 6     # cuadros de la ventana que pasan por YOLO

PRETRAINED = Path(__file__).resolve().parents[1] / "models" / "pretrained"
YOLO_WEIGHTS = PRETRAINED / "yolov8n.pt"
# pesos ImageNet originales de torchvision, re-alojados en Hugging Face (timm/resnet18.tv_in1k)
RESNET_WEIGHTS = PRETRAINED / "resnet18_tv_in1k.safetensors"
RESNET_URL = "https://huggingface.co/timm/resnet18.tv_in1k/resolve/main/model.safetensors"

_yolo = None
_cnn = None
_device = None


def device():
    global _device
    if _device is None:
        import torch
        _device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    return _device


# ----------------------------------------------------------------------------
# 1. Visuales y movimiento
# ----------------------------------------------------------------------------
def _stats(prefix: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std(),
            f"{prefix}_max": x.max(), f"{prefix}_min": x.min()}


def visual_features(win: Window) -> dict:
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in win.small]
    hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in win.small]
    feats = {}
    feats |= _stats("brightness", [g.mean() for g in gray])
    feats |= _stats("contrast", [g.std() for g in gray])
    feats |= _stats("saturation", [h[..., 1].mean() for h in hsv])
    feats |= _stats("green_ratio", [green_ratio(f) for f in win.small])
    feats |= _stats("edge_density", [(cv2.Canny(g, 100, 200) > 0).mean() for g in gray])

    diffs = np.array([cv2.absdiff(a, b).mean() for a, b in zip(gray[:-1], gray[1:])])
    feats |= _stats("motion", diffs)
    feats["motion_peak_pos"] = diffs.argmax() / max(len(diffs) - 1, 1)
    feats["shot_cut"] = float(diffs.max() > 40)  # cambio de toma dentro de la ventana

    # flujo óptico: la mediana aproxima el paneo de cámara, el residuo es el movimiento propio
    cam, resid, resid_p90, angle_conc = [], [], [], []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        med = np.median(flow.reshape(-1, 2), axis=0)
        cam.append(np.linalg.norm(med))
        r = np.linalg.norm(flow - med, axis=2)
        resid.append(r.mean())
        resid_p90.append(np.percentile(r, 90))
        ang = np.arctan2(flow[..., 1], flow[..., 0]).ravel()
        angle_conc.append(np.abs(np.exp(1j * ang).mean()))  # 1 = todo se mueve igual
    feats |= _stats("cam_motion", cam)
    feats |= _stats("player_flow", resid)
    feats["player_flow_p90_max"] = max(resid_p90)
    feats["flow_coherence_mean"] = float(np.mean(angle_conc))
    feats["cam_accel_max"] = float(np.abs(np.diff(cam)).max()) if len(cam) > 1 else 0.0
    return feats


# ----------------------------------------------------------------------------
# 2. Jugadores y balón con YOLO
# ----------------------------------------------------------------------------
def yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(YOLO_WEIGHTS) if YOLO_WEIGHTS.exists() else "yolov8n.pt")
    return _yolo


def yolo_features(win: Window) -> dict:
    sel = np.linspace(0, len(win.full) - 1, N_YOLO_FRAMES).round().astype(int)
    frames = [win.full[i] for i in sel]
    h, w = frames[0].shape[:2]
    res = yolo().predict(frames, imgsz=YOLO_WIDTH, classes=[PERSON, BALL], conf=0.15,
                         device=device(), verbose=False)
    n_players, spread_x, spread_y, ball_xy, ball_near = [], [], [], [], []
    for r in res:
        b = r.boxes
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        xy = b.xywh.cpu().numpy()[:, :2] / [w, h]
        people = xy[(cls == PERSON) & (conf >= 0.3)]
        n_players.append(len(people))
        spread_x.append(people[:, 0].std() if len(people) > 1 else 0.0)
        spread_y.append(people[:, 1].std() if len(people) > 1 else 0.0)
        balls = np.where(cls == BALL)[0]
        if len(balls):
            bxy = xy[balls[conf[balls].argmax()]]
            ball_xy.append(bxy)
            if len(people):
                ball_near.append(np.linalg.norm(people - bxy, axis=1).min())
        else:
            ball_xy.append(None)

    seen = [p for p in ball_xy if p is not None]
    speeds = [np.linalg.norm(b - a) for a, b in zip(ball_xy[:-1], ball_xy[1:])
              if a is not None and b is not None]
    return {
        "players_mean": float(np.mean(n_players)),
        "players_std": float(np.std(n_players)),
        "players_spread_x": float(np.mean(spread_x)),
        "players_spread_y": float(np.mean(spread_y)),
        "ball_visible": len(seen) / len(ball_xy),
        "ball_speed_max": float(max(speeds)) if speeds else 0.0,
        "ball_speed_mean": float(np.mean(speeds)) if speeds else 0.0,
        "ball_y_mean": float(np.mean([p[1] for p in seen])) if seen else 0.5,
        "ball_edge_dist": float(min(min(p[0], 1 - p[0], p[1], 1 - p[1]) for p in seen)) if seen else 0.5,
        "ball_player_dist": float(np.mean(ball_near)) if ball_near else 1.0,
    }


# ----------------------------------------------------------------------------
# 3. Embeddings con ResNet18
# ----------------------------------------------------------------------------
def cnn():
    global _cnn
    if _cnn is None:
        import torch
        from safetensors.torch import load_file
        from torchvision.models import resnet18
        if not RESNET_WEIGHTS.exists():
            torch.hub.download_url_to_file(RESNET_URL, str(RESNET_WEIGHTS))
        m = resnet18()
        m.load_state_dict(load_file(RESNET_WEIGHTS))
        m.fc = torch.nn.Identity()
        _cnn = m.eval().to(device())
    return _cnn


def cnn_embeddings(win: Window) -> np.ndarray:
    """(N_FRAMES, 512) float32."""
    import torch
    x = torch.from_numpy(win.cnn).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        return cnn()(((x - mean) / std).to(device())).cpu().numpy().astype(np.float32)


def window_features(win: Window) -> tuple[dict, np.ndarray]:
    """Features tabulares + secuencia de embeddings de una ventana."""
    feats = {"green_window": win.green} | visual_features(win) | yolo_features(win)
    return feats, cnn_embeddings(win)
