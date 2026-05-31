"""Treino e comparação dos 3 modelos de detecção de churn.

Modelos (selecionados via skill /model-selector):
  1. Logistic Regression — baseline interpretável (com StandardScaler).
  2. Random Forest        — robusto, captura não-linearidades.
  3. XGBoost              — teto de performance em dados tabulares.

Todos tratam o desbalanceamento de classes (~14,5% churn).
"""

from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .data import load_data
from .evaluate import compare_models
from .features import make_splits


def build_models(y_train) -> dict:
    """Instancia os 3 modelos prontos para treino."""
    # scale_pos_weight = nº negativos / nº positivos, para balancear o XGBoost.
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_all():
    """Pipeline completo: carrega dados, treina, avalia e salva os modelos."""
    df = load_data()
    X_train, X_test, y_train, y_test = make_splits(df)

    models = build_models(y_train)
    trained = {}
    for name, model in models.items():
        print(f"[treino] {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
        trained[name] = model

    results = compare_models(trained, X_test, y_test)
    print("\n=== Comparação de modelos (teste) ===")
    print(results.to_string(index=False))
    return trained, results


if __name__ == "__main__":
    train_all()
