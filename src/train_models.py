"""Entrena los modelos tabulares con ajuste de hiperparametros por partido.

Para cada modelo (regresion logistica, Random Forest, XGBoost) se hace una
busqueda en malla con GroupKFold sobre match_id y metrica F1 macro, se reentrena
con los mejores parametros sobre todo el split de entrenamiento y se guarda:

- models/<modelo>.joblib                 pipeline entrenado
- outputs/results/cv_<modelo>.csv        F1 macro por pliegue (mejores params)
- outputs/results/tuning.json            mejores params y F1 macro CV por modelo
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modeling

MODELS_DIR = Path("models")
RESULTS_DIR = Path("outputs/results")
SEED = 42

# (pipeline, malla de hiperparametros) por modelo.
SPECS = {
    "logreg": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=4000, random_state=SEED))]),
        {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    ),
    "rf": (
        Pipeline([("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"clf__n_estimators": [300, 600], "clf__max_depth": [None, 10, 20],
         "clf__min_samples_leaf": [1, 2, 4]},
    ),
    "xgb": (
        Pipeline([("clf", XGBClassifier(
            objective="multi:softprob", num_class=len(modeling.LABELS),
            tree_method="hist", eval_metric="mlogloss", random_state=SEED))]),
        {"clf__n_estimators": [300, 600], "clf__max_depth": [3, 5],
         "clf__learning_rate": [0.05, 0.1]},
    ),
}


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, _ = modeling.load_split("train")
    cv = modeling.cv_splitter(len(np.unique(groups)))
    tuning = {}

    for name, (pipe, grid) in SPECS.items():
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X, y, groups=groups)
        best = search.best_estimator_

        # F1 macro por pliegue con los mejores hiperparametros.
        fold_scores = cross_val_score(best, X, y, groups=groups,
                                      scoring="f1_macro", cv=cv)
        pd.DataFrame({"fold": range(1, len(fold_scores) + 1),
                      "f1_macro": fold_scores.round(4)}).to_csv(
            RESULTS_DIR / f"cv_{name}.csv", index=False)

        joblib.dump(best, MODELS_DIR / f"{name}.joblib")
        tuning[name] = {
            "best_params": {k: v for k, v in search.best_params_.items()},
            "cv_f1_macro": round(float(search.best_score_), 4),
            "cv_f1_macro_std": round(float(fold_scores.std()), 4),
        }
        print(f"{name}: F1 macro CV = {tuning[name]['cv_f1_macro']} "
              f"(+/- {tuning[name]['cv_f1_macro_std']})  params={search.best_params_}")

    with open(RESULTS_DIR / "tuning.json", "w") as f:
        json.dump(tuning, f, indent=2, ensure_ascii=False)
    print("Guardado en models/ y outputs/results/")


if __name__ == "__main__":
    main()
