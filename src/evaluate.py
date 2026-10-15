"""Evaluación final de los cuatro modelos en el partido de prueba retenido.

Uso desde la raíz: python src/evaluate.py

El split `valid` (partido 3c993bd2, 362 subclips) no intervino en el ajuste de ningún
modelo; aquí se usa una sola vez para medir la generalización a un partido nuevo.

Salidas en outputs/results/:
  metrics.csv              F1 macro, exactitud, precisión/recall macro, CV y tiempos por modelo
  per_class.csv            precisión, recall y F1 de cada clase por modelo
  confusion_<modelo>.csv   matriz de confusión (filas = real, columnas = predicha)
  feature_importance.csv   importancia por permutación (caída de F1 macro) de los tabulares
  test_predictions.csv     predicción y probabilidades de cada modelo por subclip
  tuning_curves.csv        F1 macro CV al variar el hiperparámetro principal de cada modelo
Figuras estáticas en outputs/20_*.png ... outputs/27_*.png
"""

import os

# PyTorch y XGBoost traen cada uno su runtime de OpenMP; en macOS, con varios hilos,
# el proceso termina en segfault. Debe fijarse antes de importar cualquiera de los dos.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support)
from sklearn.model_selection import validation_curve  # noqa: E402

import modeling  # noqa: E402
import train_temporal  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results"
SEED = 42

MODELS = ["logreg", "rf", "xgb", "gru"]
NAMES = {"logreg": "Regresión logística", "rf": "Random Forest",
         "xgb": "XGBoost", "gru": "GRU (ResNet18)"}
LABELS_ES = {"background": "Fondo", "challenge": "Disputa", "play": "Pase", "throwin": "Saque de banda"}

# Paleta validada para daltonismo (las tres clases de acción pasan todos los pares;
# el fondo, que es "sin acción", va en gris neutro). La comparten la app y el informe.
CLASS_COLORS = {"background": "#8b8a85", "challenge": "#eb6834", "play": "#2a78d6", "throwin": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6a66", "#e4e3df", "#fcfcfb"
HIGHLIGHT, NEUTRAL = "#2a78d6", "#b9b8b2"
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
    })


# ----------------------------------------------------------------------------
# Datos y predicciones
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(ROOT / modeling.FEATURES_CSV)
    with np.load(ROOT / "data/subclip_embeddings.npz", allow_pickle=True) as archive:
        if not np.array_equal(archive["files"], df["file"].to_numpy()):
            raise ValueError("El orden de los embeddings no coincide con subclip_features.csv")
        emb = archive["emb"].astype(np.float32)
    test = df["split"].eq("valid").to_numpy()
    if set(df.loc[test, "match_id"]) & set(df.loc[~test, "match_id"]):
        raise ValueError("Hay partidos compartidos entre entrenamiento y prueba")
    return df, emb, test


def timed(fn, repeats=5):
    """Mediana de `repeats` ejecuciones, en segundos, y el último resultado."""
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def predict_all(df, emb, test):
    """Probabilidades en prueba y tiempos de ajuste/inferencia de cada modelo."""
    cols = modeling.feature_columns(df)
    Xtr, Xte = df.loc[~test, cols].to_numpy(float), df.loc[test, cols].to_numpy(float)
    ytr = modeling.encode(df.loc[~test, "label"])
    proba, timing, fitted = {}, {}, {}
    torch.set_num_threads(2)

    for name in ["logreg", "rf", "xgb"]:
        model = joblib.load(ROOT / f"models/{name}.joblib")
        # tiempo de reentrenar con los mejores hiperparámetros (sin la búsqueda en malla)
        fit_s, _ = timed(lambda: clone(model).fit(Xtr, ytr), repeats=1)
        pred_s, p = timed(lambda: model.predict_proba(Xte))
        proba[name], fitted[name] = p, model
        timing[name] = {"fit_s": fit_s, "infer_us_per_clip": 1e6 * pred_s / len(Xte)}

    gru, ckpt = train_temporal.load_model()
    x_te = torch.from_numpy(emb[test])
    with torch.inference_mode():
        pred_s, logits = timed(lambda: gru(x_te))
    proba["gru"] = torch.softmax(logits, dim=1).numpy()
    fit_s, _ = timed(lambda: train_temporal.fit_model(emb[~test], ytr, ckpt["config"]["hidden_size"],
                                                      ckpt["epochs"]), repeats=1)
    timing["gru"] = {"fit_s": fit_s, "infer_us_per_clip": 1e6 * pred_s / len(x_te)}
    return proba, timing, fitted, cols


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def cv_scores():
    tuning = json.loads((RES / "tuning.json").read_text())
    gru = json.loads((RES / "cv_gru.json").read_text())
    out = {k: (v["cv_f1_macro"], v["cv_f1_macro_std"]) for k, v in tuning.items()}
    out["gru"] = (gru["cv_f1_macro"], gru["cv_f1_macro_std"])
    return out


def metric_tables(y, proba, timing):
    cv = cv_scores()
    labels = list(range(len(modeling.LABELS)))
    rows, per_class = [], []
    for name in MODELS:
        pred = proba[name].argmax(1)
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        rows.append({
            "modelo": name, "nombre": NAMES[name],
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "precision_macro": p.mean(), "recall_macro": r.mean(),
            "ap_macro": np.mean([average_precision_score(y == k, proba[name][:, k]) for k in labels]),
            "cv_f1_macro": cv[name][0], "cv_f1_macro_std": cv[name][1],
            **timing[name],
        })
        for k, lab in enumerate(modeling.LABELS):
            per_class.append({"modelo": name, "clase": lab, "precision": p[k], "recall": r[k],
                              "f1": f[k], "soporte": int(s[k])})
        cm = confusion_matrix(y, pred, labels=labels)
        pd.DataFrame(cm, index=modeling.LABELS, columns=modeling.LABELS).rename_axis("real").to_csv(
            RES / f"confusion_{name}.csv")
    metrics = pd.DataFrame(rows).round(4)
    per_class = pd.DataFrame(per_class).round(4)
    metrics.to_csv(RES / "metrics.csv", index=False)
    per_class.to_csv(RES / "per_class.csv", index=False)
    return metrics, per_class


def predictions_table(df, test, proba):
    out = df.loc[test, ["file", "match_id", "label"]].reset_index(drop=True)
    for name in MODELS:
        out[f"pred_{name}"] = modeling.decode(proba[name].argmax(1))
        for k, lab in enumerate(modeling.LABELS):
            out[f"p_{name}_{lab}"] = proba[name][:, k].round(4)
    out.to_csv(RES / "test_predictions.csv", index=False)
    return out


def importance_table(df, test, fitted, cols):
    """Permutación sobre el partido de prueba: cuánto cae el F1 macro al barajar cada variable."""
    X, y = df.loc[test, cols].to_numpy(float), modeling.encode(df.loc[test, "label"])
    rows = []
    for name, model in fitted.items():
        res = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=10,
                                     random_state=SEED, n_jobs=-1)
        rows += [{"modelo": name, "variable": c, "importancia": m, "std": s}
                 for c, m, s in zip(cols, res.importances_mean, res.importances_std)]
    imp = pd.DataFrame(rows).round(5).sort_values(["modelo", "importancia"], ascending=[True, False])
    imp.to_csv(RES / "feature_importance.csv", index=False)
    return imp


def tuning_curves(df, test):
    """Curvas de validación por partido (solo train) del hiperparámetro principal."""
    cols = modeling.feature_columns(df)
    X, y = df.loc[~test, cols].to_numpy(float), modeling.encode(df.loc[~test, "label"])
    groups = df.loc[~test, "match_id"].to_numpy()
    cv = modeling.cv_splitter(len(np.unique(groups)))
    grids = {"logreg": ("clf__C", [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]),
             "rf": ("clf__max_depth", [2, 3, 5, 8, 10, 15, 20, 30]),
             "xgb": ("clf__n_estimators", [50, 100, 200, 300, 400, 600, 800])}
    rows = []
    for name, (param, values) in grids.items():
        model = joblib.load(ROOT / f"models/{name}.joblib")
        tr, va = validation_curve(model, X, y, param_name=param, param_range=values, groups=groups,
                                  cv=cv, scoring="f1_macro", n_jobs=-1)
        rows += [{"modelo": name, "parametro": param.split("__")[1], "valor": v,
                  "f1_train": a.mean(), "f1_cv": b.mean(), "f1_cv_std": b.std()}
                 for v, a, b in zip(values, tr, va)]
    gru = json.loads((RES / "cv_gru.json").read_text())
    rows += [{"modelo": "gru", "parametro": f"hidden={c['hidden_size']}", "valor": c["epochs"],
              "f1_train": np.nan, "f1_cv": c["mean_f1_macro"], "f1_cv_std": c["std_f1_macro"]}
             for c in gru["candidates"]]
    curves = pd.DataFrame(rows).round(4)
    curves.to_csv(RES / "tuning_curves.csv", index=False)
    return curves


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def fig_f1_models(metrics):
    m = metrics.set_index("modelo").loc[MODELS]
    best = m["f1_macro"].idxmax()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(MODELS))
    ax.bar(x - 0.19, m["cv_f1_macro"], 0.36, color=NEUTRAL, label="Validación cruzada (7 partidos de train)",
           yerr=m["cv_f1_macro_std"], error_kw={"ecolor": MUTED, "lw": 1, "capsize": 3})
    colors = [HIGHLIGHT if k == best else "#5b5a55" for k in MODELS]
    ax.bar(x + 0.19, m["f1_macro"], 0.36, color=colors, label="Partido de prueba (no visto)")
    for xi, v in zip(x, m["f1_macro"]):
        ax.text(xi + 0.19, v + 0.01, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.axhline(0.25, color=MUTED, ls="--", lw=1)
    ax.text(len(MODELS) - 0.5, 0.255, "azar (4 clases)", ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.set_xticks(x, [NAMES[k] for k in MODELS])
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, max(0.6, m["f1_macro"].max() + 0.1))
    ax.set_title("F1 macro por modelo: validación cruzada vs. partido de prueba")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "20_f1_por_modelo.png")


def fig_f1_classes(per_class):
    tab = per_class.pivot(index="modelo", columns="clase", values="f1").loc[MODELS, modeling.LABELS]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(tab.to_numpy(), cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else INK, fontsize=10)
    ax.set_xticks(range(4), [LABELS_ES[c] for c in modeling.LABELS])
    ax.set_yticks(range(len(MODELS)), [NAMES[k] for k in MODELS])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="F1")
    ax.set_title("F1 por clase en el partido de prueba")
    save(fig, "21_f1_por_clase.png")


def fig_confusion(y, proba):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    ticks = [LABELS_ES[c] for c in modeling.LABELS]
    for ax, name in zip(axes, MODELS):
        cm = confusion_matrix(y, proba[name].argmax(1), labels=range(4))
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=SEQ_CMAP, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), ticks, rotation=30, ha="right")
        ax.set_yticks(range(4), ticks if ax is axes[0] else [])
        ax.set_xlabel("Predicha")
        ax.grid(False)
        ax.set_title(NAMES[name], fontsize=11)
    axes[0].set_ylabel("Real")
    fig.suptitle("Matrices de confusión en el partido de prueba (conteo y % por fila)",
                 fontweight="bold", color=INK)
    save(fig, "22_matrices_confusion.png")


def fig_time_f1(metrics):
    m = metrics.set_index("modelo").loc[MODELS]
    best = m["f1_macro"].idxmax()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, r in m.iterrows():
        ax.scatter(r["infer_us_per_clip"], r["f1_macro"], s=60 + 25 * np.log1p(r["fit_s"]) ** 2,
                   color=HIGHLIGHT if k == best else "#5b5a55", edgecolor=SURFACE, lw=2, zorder=3)
        ax.annotate(f"{NAMES[k]}\najuste {r['fit_s']:.1f} s", (r["infer_us_per_clip"], r["f1_macro"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de inferencia por subclip (µs, escala log) — sin contar la extracción de variables")
    ax.set_ylabel("F1 macro en prueba")
    ax.set_ylim(0, max(0.5, m["f1_macro"].max() + 0.08))
    ax.set_title("Costo computacional vs. desempeño (tamaño = tiempo de ajuste)")
    save(fig, "23_tiempo_vs_f1.png")


def fig_importance(imp, best):
    top = imp[imp["modelo"] == best].nlargest(15, "importancia").iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = [HIGHLIGHT if v > 0 else NEUTRAL for v in top["importancia"]]
    ax.barh(top["variable"], top["importancia"], xerr=top["std"], color=colors,
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Caída del F1 macro al permutar la variable (partido de prueba)")
    ax.set_title(f"Variables más importantes — {NAMES[best]}")
    ax.grid(axis="y", visible=False)
    save(fig, "24_importancia_variables.png")


def fig_pr(y, proba, best):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for k, lab in enumerate(modeling.LABELS):
        p, r, _ = precision_recall_curve(y == k, proba[best][:, k])
        ap = average_precision_score(y == k, proba[best][:, k])
        prev = (y == k).mean()
        ax.plot(r, p, color=CLASS_COLORS[lab], lw=2, label=f"{LABELS_ES[lab]} (AP {ap:.2f}, azar {prev:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Curvas precisión-recall por clase — {NAMES[best]}")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, "25_curvas_pr.png")


def fig_tuning(curves, tuning):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    labels = {"logreg": "C (inverso de la regularización)", "rf": "max_depth",
              "xgb": "n_estimators (learning_rate 0.05, max_depth 3)"}
    for ax, name in zip(axes[:3], ["logreg", "rf", "xgb"]):
        c = curves[curves["modelo"] == name].astype({"valor": float})
        ax.plot(c["valor"], c["f1_train"], color=NEUTRAL, lw=2, marker="o", ms=5, label="Entrenamiento")
        ax.plot(c["valor"], c["f1_cv"], color=HIGHLIGHT, lw=2, marker="o", ms=5, label="CV por partido")
        ax.fill_between(c["valor"], c["f1_cv"] - c["f1_cv_std"], c["f1_cv"] + c["f1_cv_std"],
                        color=HIGHLIGHT, alpha=0.12, lw=0)
        best_val = tuning[name]["best_params"].get(f"clf__{c['parametro'].iloc[0]}")
        if best_val is not None:
            ax.axvline(float(best_val), color=MUTED, ls="--", lw=1)
        else:  # max_depth=None: árboles sin límite de profundidad
            ax.text(0.97, 0.05, "elegido: sin límite (None)", transform=ax.transAxes, ha="right",
                    color=MUTED, fontsize=8.5)
        if name == "logreg":
            ax.set_xscale("log")
        ax.set_xlabel(labels[name])
        ax.set_title(NAMES[name], fontsize=11)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("F1 macro")
    axes[0].legend(fontsize=8, loc="upper left")
    g = curves[curves["modelo"] == "gru"]
    x = np.arange(len(g))
    ax = axes[3]
    ax.bar(x, g["f1_cv"], yerr=g["f1_cv_std"], color=[HIGHLIGHT if v == g["f1_cv"].max() else NEUTRAL
                                                    for v in g["f1_cv"]],
           error_kw={"ecolor": MUTED, "lw": 1, "capsize": 3})
    ax.set_xticks(x, [f"{p}\n{int(v)} ép." for p, v in zip(g["parametro"], g["valor"])], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(NAMES["gru"], fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Ajuste de hiperparámetros: F1 macro en validación cruzada por partido (línea punteada = elegido)",
                 fontweight="bold", color=INK)
    save(fig, "26_ajuste_hiperparametros.png")


def fig_eda_link(df, test):
    """Las variables que más usa el mejor modelo, por clase, y el cambio de dominio del partido de prueba."""
    variables = [("cam_motion_mean", "Paneo de cámara medio"), ("player_flow_mean", "Flujo propio de jugadores"),
                 ("players_spread_y", "Dispersión vertical de jugadores")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for ax, (col, title) in zip(axes, variables):
        data = [df.loc[df["label"] == lab, col] for lab in modeling.LABELS]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                        medianprops={"color": INK, "lw": 1.5})
        for patch, lab in zip(bp["boxes"], modeling.LABELS):
            patch.set(facecolor=CLASS_COLORS[lab], alpha=0.75, edgecolor=SURFACE)
        ax.set_xticks(range(1, 5), [LABELS_ES[c] for c in modeling.LABELS], rotation=20)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
    g = df.groupby("match_id")["green_window"].mean().sort_values()
    test_match = df.loc[test, "match_id"].iloc[0]
    axes[3].barh(g.index, g.values, color=[HIGHLIGHT if m == test_match else NEUTRAL for m in g.index])
    axes[3].set_title("Césped visible por partido (azul = prueba)", fontsize=11)
    axes[3].set_xlabel("Cobertura media de césped")
    axes[3].grid(axis="y", visible=False)
    fig.suptitle("Del análisis exploratorio al modelo: variables más útiles por clase y cambio de dominio",
                 fontweight="bold", color=INK)
    save(fig, "27_variables_por_clase.png")


def main():
    style()
    RES.mkdir(parents=True, exist_ok=True)
    df, emb, test = load_data()
    y = modeling.encode(df.loc[test, "label"])
    proba, timing, fitted, cols = predict_all(df, emb, test)
    metrics, per_class = metric_tables(y, proba, timing)
    predictions_table(df, test, proba)
    best_tab = metrics[metrics["modelo"] != "gru"].sort_values("f1_macro").iloc[-1]["modelo"]
    imp = importance_table(df, test, fitted, cols)
    curves = tuning_curves(df, test)

    fig_f1_models(metrics)
    fig_f1_classes(per_class)
    fig_confusion(y, proba)
    fig_time_f1(metrics)
    fig_importance(imp, best_tab)
    fig_pr(y, proba, best_tab)
    fig_tuning(curves, json.loads((RES / "tuning.json").read_text()))
    fig_eda_link(df, test)

    print(metrics[["modelo", "f1_macro", "accuracy", "cv_f1_macro", "fit_s", "infer_us_per_clip"]]
          .to_string(index=False))
    print("\nF1 por clase:\n", per_class.pivot(index="modelo", columns="clase", values="f1").round(3))
    print("\nFiguras 20-27 en outputs/ y tablas en outputs/results/")


if __name__ == "__main__":
    main()
