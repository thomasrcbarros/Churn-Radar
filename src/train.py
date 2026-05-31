"""Treino e comparação dos 3 modelos de detecção de churn.

Modelos (selecionados via skill /model-selector):
  1. Logistic Regression — baseline interpretável (com StandardScaler).
  2. Random Forest        — robusto, captura não-linearidades.
  3. XGBoost              — teto de performance em dados tabulares.

Desbalanceamento (~14,5% churn) tratado de duas formas comparáveis:
  - baseline: pesos de classe (`class_weight` / `scale_pos_weight`);
  - SMOTE: oversampling sintético da classe minoritária, aplicado SÓ no treino
    (via Pipeline do imbalanced-learn, dentro do `.fit`).
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config
from .data import load_data
from .evaluate import compare_models
from .features import make_splits


def build_models(y_train, use_smote: bool = False) -> dict:
    """Instancia os 3 modelos prontos para treino.

    Se ``use_smote`` for True, o balanceamento por pesos é desativado e cada
    estimador é embrulhado numa Pipeline do imbalanced-learn que aplica SMOTE
    apenas nas partições de treino (nunca no teste).
    """
    # scale_pos_weight = nº negativos / nº positivos, para balancear o XGBoost.
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # Com SMOTE não usamos pesos de classe (evita corrigir o desbalanceamento
    # duas vezes); sem SMOTE mantemos os pesos balanceados.
    class_weight = None if use_smote else "balanced"
    xgb_spw = 1.0 if use_smote else scale_pos_weight

    logreg = LogisticRegression(
        max_iter=1000, class_weight=class_weight, random_state=config.RANDOM_STATE
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=xgb_spw,
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    if use_smote:
        smote = SMOTE(random_state=config.RANDOM_STATE)
        return {
            # Scaler antes do SMOTE: distâncias do SMOTE ficam em escala comparável.
            "LogisticRegression": ImbPipeline(
                [("scaler", StandardScaler()), ("smote", smote), ("clf", logreg)]
            ),
            "RandomForest": ImbPipeline([("smote", smote), ("clf", rf)]),
            "XGBoost": ImbPipeline([("smote", smote), ("clf", xgb)]),
        }

    return {
        "LogisticRegression": Pipeline(
            steps=[("scaler", StandardScaler()), ("clf", logreg)]
        ),
        "RandomForest": rf,
        "XGBoost": xgb,
    }


def train_all(use_smote: bool = False):
    """Pipeline completo: carrega dados, treina, avalia e salva os modelos."""
    df = load_data()
    X_train, X_test, y_train, y_test = make_splits(df)

    suffix = "_smote" if use_smote else ""
    models = build_models(y_train, use_smote=use_smote)
    trained = {}
    for name, model in models.items():
        print(f"[treino] {name}{suffix}...")
        model.fit(X_train, y_train)
        joblib.dump(model, config.MODELS_DIR / f"{name}{suffix}.joblib")
        trained[name] = model

    results = compare_models(trained, X_test, y_test)
    label = "com SMOTE" if use_smote else "baseline (pesos de classe)"
    print(f"\n=== Comparação de modelos — {label} (teste) ===")
    print(results.to_string(index=False))
    return trained, results


def compare_resampling():
    """Treina baseline vs SMOTE e mostra o efeito no recall lado a lado."""
    df = load_data()
    X_train, X_test, y_train, y_test = make_splits(df)

    combined = []
    for use_smote in (False, True):
        models = build_models(y_train, use_smote=use_smote)
        for name, model in models.items():
            model.fit(X_train, y_train)
        res = compare_models(models, X_test, y_test)
        res.insert(1, "Strategy", "SMOTE" if use_smote else "baseline")
        combined.append(res)

    table = pd.concat(combined, ignore_index=True)
    table = table.sort_values(["Model", "Strategy"]).reset_index(drop=True)
    print("\n=== Baseline vs SMOTE (foco em Recall) ===")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_resampling()
    else:
        train_all(use_smote="smote" in sys.argv[1:])
