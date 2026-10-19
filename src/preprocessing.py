"""Preprocesamiento de video compartido por el entrenamiento y la aplicación.

Todo clip (subclip etiquetado o video nuevo subido en la app) pasa por las mismas
funciones: se lee con OpenCV, se toma una ventana fija que termina justo después del
momento a clasificar (BEFORE_S antes y AFTER_S después), se muestrean N_FRAMES cuadros
equiespaciados, se recorta la franja del marcador y se redimensiona. Así el modelo nunca
ve la duración del clip, que en los subclips cambia mucho entre clases.

Los subclips etiquetados se cortaron sin recodificar: el mp4 trae desde el keyframe
anterior al evento (0-10 s de contexto) pero una edit list oculta todo lo previo al
evento. Le pedimos a FFmpeg que ignore la edit list para recuperar ese contexto; el
evento queda entonces en el cuadro n_frames - (next_s - event_s) * fps.
"""
from __future__ import annotations

import os

# debe definirse antes de abrir cualquier VideoCapture
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "ignore_editlist;1")
if os.name == "nt":
    # en Windows el backend FFmpeg de OpenCV lee el entorno de msvcrt, que no ve os.environ
    import ctypes
    ctypes.cdll.msvcrt._putenv(
        f"OPENCV_FFMPEG_CAPTURE_OPTIONS={os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']}".encode())

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

BEFORE_S = 1.5          # contexto antes del momento a clasificar
AFTER_S = 0.12          # después: los subclips solo garantizan 0.15 s tras el evento
N_FRAMES = 16           # cuadros por ventana (~10 fps efectivos)
SMALL_SIZE = (320, 180)  # para features clásicas y flujo óptico
CNN_SIZE = (224, 224)    # entrada de la CNN preentrenada
YOLO_WIDTH = 1280        # ancho para YOLO (el balón es muy pequeño en 1080p)
TOP_CROP = 0.10          # franja superior con marcador y marca de agua
BOTTOM_CROP = 0.04

# máscara de césped usada en el EDA (HSV de OpenCV)
GREEN_LO = np.array([35, 40, 40])
GREEN_HI = np.array([90, 255, 255])
MIN_GREEN_RATIO = 0.30  # debajo de esto la toma no muestra la cancha (primer plano, público)


@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def window_indices(info: VideoInfo, t_s: float, n: int = N_FRAMES) -> np.ndarray:
    """Índices de N cuadros equiespaciados en [t_s - BEFORE_S, t_s + AFTER_S].

    Si falta contexto al inicio se repite el primer cuadro (no se desplaza la ventana),
    de modo que el momento a clasificar siempre cae en la misma posición.
    """
    last = max(info.n_frames - 1, 0)
    idx = np.linspace((t_s - BEFORE_S) * info.fps, (t_s + AFTER_S) * info.fps, n)
    return np.clip(np.round(idx), 0, last).astype(int)


def subclip_event_time(info: VideoInfo, event_s: float, next_s: float) -> float:
    """Segundo del subclip (con edit list ignorada) en que ocurre el evento."""
    return max(info.n_frames - 1 - (next_s - event_s) * info.fps, 0) / info.fps


def read_frames(path: str | Path, indices: np.ndarray) -> list[np.ndarray]:
    """Lee los cuadros pedidos (BGR, resolución original) de forma secuencial."""
    cap = cv2.VideoCapture(str(path))
    wanted = sorted(set(int(i) for i in indices))
    frames: dict[int, np.ndarray] = {}
    pos = 0
    for idx in wanted:
        # avance secuencial con grab (sin decodificar): el seek de OpenCV falla en algunos mp4
        while pos < idx:
            if not cap.grab():
                break
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break
        frames[idx] = frame
    cap.release()
    if not frames:
        raise ValueError(f"No se pudo leer ningún cuadro de {path}")
    last_ok = max(frames)
    return [frames.get(int(i), frames[last_ok]) for i in indices]


def crop_overlay(frame: np.ndarray) -> np.ndarray:
    """Quita la franja del marcador y la marca de agua."""
    h = frame.shape[0]
    return frame[int(h * TOP_CROP): int(h * (1 - BOTTOM_CROP))]


def green_ratio(frame_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(cv2.inRange(hsv, GREEN_LO, GREEN_HI).mean() / 255)


@dataclass
class Window:
    """Ventana preprocesada lista para extraer features."""
    full: list[np.ndarray]    # BGR recortado a resolución original (para YOLO)
    small: list[np.ndarray]   # BGR 320x180
    cnn: np.ndarray           # (N, 224, 224, 3) RGB uint8
    green: float              # cobertura media de césped
    start_s: float
    end_s: float

    @property
    def is_field_view(self) -> bool:
        return self.green >= MIN_GREEN_RATIO


def preprocess_window(path: str | Path, t_s: float, info: VideoInfo | None = None) -> Window:
    """Prepara la ventana que termina AFTER_S segundos después de t_s."""
    info = info or video_info(path)
    idx = window_indices(info, t_s)
    raw = [crop_overlay(f) for f in read_frames(path, idx)]
    small = [cv2.resize(f, SMALL_SIZE, interpolation=cv2.INTER_AREA) for f in raw]
    cnn = np.stack([cv2.cvtColor(cv2.resize(f, CNN_SIZE, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB) for f in raw])
    return Window(full=raw, small=small, cnn=cnn,
                  green=float(np.mean([green_ratio(f) for f in small])),
                  start_s=idx[0] / info.fps, end_s=idx[-1] / info.fps)


def sliding_times(info: VideoInfo, stride_s: float = 0.5) -> np.ndarray:
    """Momentos a clasificar al recorrer un video completo en la app."""
    first, last = BEFORE_S, info.duration_s - AFTER_S
    if last <= first:
        return np.array([max(last, 0.0)])
    return np.arange(first, last + 1e-6, stride_s)


def parse_subclip_name(name: str) -> dict:
    """'1606b0e6_0_1130388_1130754.mp4' -> video_id, match_id y tiempos en segundos."""
    parts = Path(name).stem.split("_")
    return {
        "video_id": f"{parts[0]}_{parts[1]}",
        "match_id": parts[0],
        "event_s": int(parts[2]) / 1000,
        "next_s": int(parts[3]) / 1000,
    }
