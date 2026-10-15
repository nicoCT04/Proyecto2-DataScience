"""Entrena una GRU sobre embeddings de ResNet18 con validación por partido.

Uso desde la raíz: python src/train_temporal.py
Selecciona tamaño oculto y épocas mediante F1 macro en GroupKFold, usando solo
el split train. El split valid queda reservado para la evaluación final del equipo.
Guarda pesos y normalización en models/gru.pt y la búsqueda en cv_gru.json.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import modeling
import preprocessing

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.3


class GRUClassifier(nn.Module):
    """Secuencia (lote, cuadros, 512) -> logits de las cuatro clases.

    La normalización forma parte del modelo y se aprende únicamente del conjunto
    usado para entrenarlo. En inferencia basta con pasar los embeddings sin escalar.
    """

    def __init__(self, input_size=512, hidden_size=64, num_classes=4, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("scale", torch.ones(input_size))
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))

    def forward(self, x):
        _, hidden = self.gru((x - self.mean) / self.scale)
        return self.classifier(hidden[-1])


def load_training_data():
    """Valida correspondencia CSV/NPZ y devuelve solo las secuencias de train."""
    df = pd.read_csv(ROOT / modeling.FEATURES_CSV)
    # build_features guarda files como object; el archivo es un artefacto local confiable.
    with np.load(ROOT / "data/subclip_embeddings.npz", allow_pickle=True) as archive:
        if not np.array_equal(archive["files"], df["file"].to_numpy()):
            raise ValueError("El orden de los embeddings no coincide con subclip_features.csv")
        emb = archive["emb"].astype(np.float32)
    if emb.shape != (len(df), preprocessing.N_FRAMES, 512) or not np.isfinite(emb).all():
        raise ValueError("Se esperaban embeddings finitos con forma (clips, 16, 512)")
    train = df["split"].eq("train").to_numpy()
    if set(df.loc[train, "match_id"]) & set(df.loc[~train, "match_id"]):
        raise ValueError("Hay partidos compartidos entre entrenamiento y prueba")
    return emb[train], modeling.encode(df.loc[train, "label"]), df.loc[train, "match_id"].to_numpy()


def fit_model(x, y, hidden_size, epochs):
    """Entrenamiento reproducible en CPU; sin acceso al partido retenido."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = GRUClassifier(hidden_size=hidden_size)
    inputs = torch.from_numpy(x)
    model.mean.copy_(inputs.mean(dim=(0, 1)))
    model.scale.copy_(inputs.std(dim=(0, 1), correction=0).clamp_min(1e-6))
    loader = DataLoader(TensorDataset(inputs, torch.from_numpy(y).long()),
                        batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def load_model(path=None):
    """Carga portable para evaluación/app: devuelve (modelo en eval, checkpoint)."""
    checkpoint = torch.load(path or ROOT / "models/gru.pt", map_location="cpu", weights_only=True)
    model = GRUClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    x, y, groups = load_training_data()
    folds = list(modeling.cv_splitter(len(np.unique(groups))).split(x, y, groups))
    candidates = []
    for hidden_size in (32, 64):
        for epochs in (20, 40):
            scores = []
            for train, validation in folds:
                model = fit_model(x[train], y[train], hidden_size, epochs)
                with torch.inference_mode():
                    pred = model(torch.from_numpy(x[validation])).argmax(dim=1).numpy()
                scores.append(float(f1_score(y[validation], pred,
                                            labels=range(len(modeling.LABELS)),
                                            average="macro", zero_division=0)))
            candidate = {"hidden_size": hidden_size, "epochs": epochs,
                         "fold_f1_macro": scores, "mean_f1_macro": float(np.mean(scores)),
                         "std_f1_macro": float(np.std(scores))}
            candidates.append(candidate)
            print(f"hidden={hidden_size}, epochs={epochs}: F1 macro={np.mean(scores):.4f}", flush=True)

    best = max(candidates, key=lambda item: item["mean_f1_macro"])
    model = fit_model(x, y, best["hidden_size"], best["epochs"])
    config = {"input_size": x.shape[2], "hidden_size": best["hidden_size"],
              "num_classes": len(modeling.LABELS), "dropout": DROPOUT}
    checkpoint = {
        "state_dict": model.state_dict(), "config": config, "labels": modeling.LABELS,
        "n_frames": x.shape[1], "seed": SEED, "epochs": best["epochs"],
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE,
        "preprocessing": {"before_s": preprocessing.BEFORE_S, "after_s": preprocessing.AFTER_S,
                          "top_crop": preprocessing.TOP_CROP, "bottom_crop": preprocessing.BOTTOM_CROP},
    }
    results = {
        "seed": SEED, "torch_version": str(torch.__version__), "device": "cpu",
        "training_samples": len(y), "training_matches": sorted(set(groups)),
        "labels": modeling.LABELS, "sequence_shape": list(x.shape[1:]),
        "selection_metric": "f1_macro", "selection_split": "train",
        "cv": "GroupKFold por match_id", "n_splits": len(folds),
        "folds": [{"fold": i, "train_matches": sorted(set(groups[t])),
                   "validation_matches": sorted(set(groups[v])),
                   "train_samples": len(t), "validation_samples": len(v)}
                  for i, (t, v) in enumerate(folds, 1)],
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "dropout": DROPOUT,
        "weight_decay": 1e-4, "normalization": "media y desviación por dimensión, solo entrenamiento de cada pliegue",
        "candidates": candidates, "best_params": {"hidden_size": best["hidden_size"], "epochs": best["epochs"]},
        "cv_f1_macro": best["mean_f1_macro"], "cv_f1_macro_std": best["std_f1_macro"],
        "fold_f1_macro": best["fold_f1_macro"],
        "note": "CV usada para selección; no es una estimación independiente. No se evaluó el partido de prueba.",
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "outputs/results").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, ROOT / "models/gru.pt")
    (ROOT / "outputs/results/cv_gru.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print("Guardado: models/gru.pt y outputs/results/cv_gru.json")


if __name__ == "__main__":
    main()
