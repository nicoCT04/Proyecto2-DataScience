"""Extrae las features de todos los subclips etiquetados.

Salidas:
  data/subclip_features.csv    una fila por subclip: id, clase, split, partido + features tabulares
  data/subclip_embeddings.npz  secuencias (n, 16, 512) de ResNet18 en float16, mismo orden que el CSV
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import window_features
from preprocessing import preprocess_window

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    index = pd.read_csv(ROOT / "data" / "subclips_index.csv")
    rows, seqs = [], []
    t0 = time.time()
    for i, r in enumerate(index.itertuples(), 1):
        win = preprocess_window(ROOT / "data" / "subclips" / r.file, r.event_t_s)
        feats, emb = window_features(win)
        rows.append({"file": r.file, "split": r.split, "label": r.label,
                     "video_id": r.video_id, "match_id": r.match_id,
                     "duration_s": r.duration_s, "lead_s": r.event_t_s} | feats)
        seqs.append(emb.astype(np.float16))
        if i % 50 == 0:
            print(f"{i}/{len(index)}  {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows)
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(5)
    df.to_csv(ROOT / "data" / "subclip_features.csv", index=False)
    np.savez_compressed(ROOT / "data" / "subclip_embeddings.npz",
                        emb=np.stack(seqs), files=df.file.values)
    print(f"Listo: {len(df)} subclips, {df.shape[1] - 7} features, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
