"""Prepara los videos de ejemplo de la aplicación -> app/samples/

Toma un subclip por clase del partido de prueba (nunca visto en entrenamiento), lo
descarga de Hugging Face (ManuD/dfl_clips) si no está en data/subclips/ y lo copia con
su nombre original, que la app usa para ubicar el momento del evento.

Criterio de selección (documentado en la app): por cada clase, el subclip que XGBoost
clasifica correctamente con mayor probabilidad; sirven como demostración del flujo,
no como muestra representativa del desempeño (ese está en la pestaña de rendimiento).
También guarda app/samples/samples.csv con la etiqueta real de cada ejemplo.

Uso desde la raíz (después de src/evaluate.py): python src/make_app_samples.py
"""
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "app" / "samples"
BASE = "https://huggingface.co/datasets/ManuD/dfl_clips/resolve/main/subclips"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(ROOT / "outputs/results/test_predictions.csv")
    pred["p_true"] = [r[f"p_xgb_{r['label']}"] for _, r in pred.iterrows()]
    ok = pred[pred["pred_xgb"] == pred["label"]]
    chosen = ok.sort_values("p_true", ascending=False).groupby("label").head(1).sort_values("label")

    rows = []
    for r in chosen.itertuples():
        name = Path(r.file).name
        local, out = ROOT / "data" / "subclips" / r.file, DEST / name
        if local.exists():
            shutil.copy(local, out)
        elif not out.exists():
            print(f"Descargando {r.file} ...", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{r.file}", out)
        rows.append({"file": name, "label": r.label, "match_id": r.match_id})
    pd.DataFrame(rows).to_csv(DEST / "samples.csv", index=False)
    size = sum(p.stat().st_size for p in DEST.glob("*.mp4")) / 1e6
    print(f"{len(rows)} ejemplos en {DEST.relative_to(ROOT)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
