"""Utilidades compartidas para el clasificador tabular de acciones.

Clasifica cada subclip (ventana de video) en una de cuatro acciones a partir de
las variables agregadas en data/subclip_features.csv. Lo usan el entrenamiento
(train_models.py) y la evaluacion/inferencia (evaluate.py, inference.py).

El ajuste y la validacion se hacen por partido (GroupKFold sobre match_id) para
medir de forma honesta la generalizacion a un partido no visto.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

FEATURES_CSV = "data/subclip_features.csv"
META_COLS = ["file", "split", "label", "video_id", "match_id"]
# Duracion del subclip y contexto previo al evento: dependen de como se recorto el
# clip (los de background duran ~10 s y el resto ~6 s), no de la accion, y no existen
# para un video nuevo. Se excluyen para no filtrar la etiqueta.
LEAKY_COLS = ["duration_s", "lead_s"]
LABELS = ["background", "challenge", "play", "throwin"]
N_SPLITS = 5


def feature_columns(df):
    """Columnas numericas de entrada (sin metadatos ni variables del recorte)."""
    return [c for c in df.columns if c not in META_COLS + LEAKY_COLS]


def load_split(split, path=FEATURES_CSV):
    """Carga un split ('train' o 'valid') y devuelve X, y (codificada), grupos y df."""
    df = pd.read_csv(path)
    df = df[df["split"] == split].reset_index(drop=True)
    X = df[feature_columns(df)].to_numpy(dtype=float)
    y = encode(df["label"])
    groups = df["match_id"].to_numpy()
    return X, y, groups, df


def encode(labels):
    """Mapea las etiquetas de texto a enteros segun el orden fijo de LABELS."""
    index = {name: i for i, name in enumerate(LABELS)}
    return np.array([index[v] for v in labels])


def decode(codes):
    """Inverso de encode: de enteros a etiquetas de texto."""
    return np.array([LABELS[i] for i in codes])


def cv_splitter(n_groups):
    """GroupKFold acotado al numero de partidos disponibles."""
    return GroupKFold(n_splits=min(N_SPLITS, n_groups))
