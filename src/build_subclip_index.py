"""Índice de los subclips etiquetados -> data/subclips_index.csv

Una fila por subclip con su clase, partido, duración, el segundo del subclip en que
ocurre el evento (event_t_s) que es también el contexto disponible antes de él. También
guarda el evento de train.csv más cercano, para comprobar que la etiqueta del dataset
coincide con las anotaciones originales de la competencia.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import parse_subclip_name, subclip_event_time, video_info

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "data" / "subclips"
CLASSES = ["background", "challenge", "play", "throwin"]


def nearest_event(train: pd.DataFrame, video_id: str, t: float) -> tuple[str, float]:
    ev = train[(train.video_id == video_id) & train.event.isin(["play", "challenge", "throwin"])]
    if ev.empty:
        return "", np.nan
    i = (ev.time - t).abs().idxmin()
    return ev.at[i, "event"], float(ev.at[i, "time"] - t)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    rows = []
    for path in sorted(SUB.glob("*/*/*.mp4")):
        split, label = path.parts[-3], path.parts[-2]
        info = video_info(path)
        meta = parse_subclip_name(path.name)
        ev, gap = nearest_event(train, meta["video_id"], meta["event_s"])
        rows.append({
            "file": f"{split}/{label}/{path.name}", "split": split, "label": label,
            **meta, "n_frames": info.n_frames, "fps": info.fps,
            "duration_s": round(info.duration_s, 2),
            "event_t_s": round(subclip_event_time(info, meta["event_s"], meta["next_s"]), 3),
            "nearest_event": ev, "nearest_gap_s": round(gap, 3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "subclips_index.csv", index=False)

    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    lab = df[df.label != "background"]
    ok = (lab.label == lab.nearest_event) & (lab.nearest_gap_s.abs() < 0.5)
    print("Contexto previo al evento (s):", df.event_t_s.describe().round(2).to_dict())
    print(f"Etiquetas que coinciden con train.csv (±0.5 s): {ok.mean():.1%} de {len(lab)}")
    print("Partidos train:", sorted(df[df.split == "train"].match_id.unique()))
    print("Partidos valid:", sorted(df[df.split == "valid"].match_id.unique()))


if __name__ == "__main__":
    main()
