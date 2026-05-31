"""Tuning de hiperparâmetros por validação cruzada (RandomizedSearchCV).

Otimiza para PR-AUC (``average_precision``), a métrica mais adequada em dados
desbalanceados como churn. Usa StratifiedKFold para preservar a proporção de
classes em cada fold e faixas de busca voltadas a conter overfitting.

Uso:
    python -m src.tune            # busca, reporta CV + teste e salva *_tuned.joblib
    python -m src.tune --n-iter 40
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .data import load_data
from .evaluate import compare_models, overfitting_report
from .features import make_splits

SCORING = "average_precision"  # PR-AUC


def search_spaces(scale_pos_weight: float) -> dict:
    """Retorna, por modelo, (estimador, distribuição de hiperparâmetros).

    Para o Pipeline da Logistic Regression os parâmetros usam o prefixo ``clf__``.
    """
    return {
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=config.RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "clf__C": loguniform(1e-3, 1e2),
                "clf__penalty": ["l1", "l2"],
                "clf__solver": ["liblinear"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": randint(150, 500),
                "max_depth": randint(3, 12),
                "min_samples_leaf": randint(10, 50),
                "max_features": ["sqrt", "log2", 0.5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "n_estimators": randint(100, 400),
                "max_depth": randint(2, 6),
                "learning_rate": loguniform(1e-2, 2e-1),
                "subsample": uniform(0.6, 0.4),       # [0.6, 1.0]
                "colsample_bytree": uniform(0.6, 0.4),  # [0.6, 1.0]
                "min_child_weight": randint(1, 10),
                "gamma": uniform(0.0, 3.0),
                "reg_alpha": loguniform(1e-3, 5.0),
                "reg_lambda": loguniform(1e-1, 1e1),
            },
        ),
    }


def tune_all(n_iter: int = 30, cv_folds: int = 5):
    """Busca hiperparâmetros para os 3 modelos e avalia no conjunto de teste."""
    df = load_data()
    X_train, X_test, y_train, y_test = make_splits(df)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE
    )

    best_models = {}
    summary = []
    for name, (estimator, space) in search_spaces(scale_pos_weight).items():
        print(f"[tuning] {name} — RandomizedSearchCV ({n_iter} iterações, {cv_folds}-fold)...")
        search = RandomizedSearchCV(
            estimator,
            param_distributions=space,
            n_iter=n_iter,
            scoring=SCORING,
            cv=cv,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_models[name] = best
        joblib.dump(best, config.MODELS_DIR / f"{name}_tuned.joblib")
        summary.append(
            {
                "Model": name,
                "CV_PR_AUC": round(float(search.best_score_), 4),
                "best_params": search.best_params_,
            }
        )

    print("\n=== Melhores hiperparâmetros (CV, PR-AUC) ===")
    for row in summary:
        print(f"\n{row['Model']}  CV_PR_AUC={row['CV_PR_AUC']}")
        for k, v in row["best_params"].items():
            val = round(v, 4) if isinstance(v, float) else v
            print(f"   {k}: {val}")

    print("\n=== Desempenho dos modelos tunados (teste) ===")
    print(compare_models(best_models, X_test, y_test).to_string(index=False))

    print("\n=== Diagnóstico de overfitting (treino vs teste) ===")
    print(
        overfitting_report(best_models, X_train, y_train, X_test, y_test).to_string(
            index=False
        )
    )
    return best_models, pd.DataFrame(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning por validação cruzada.")
    parser.add_argument("--n-iter", type=int, default=30,
                        help="Nº de combinações amostradas por modelo (default: 30).")
    parser.add_argument("--cv", type=int, default=5,
                        help="Nº de folds da validação cruzada (default: 5).")
    args = parser.parse_args()
    tune_all(n_iter=args.n_iter, cv_folds=args.cv)
