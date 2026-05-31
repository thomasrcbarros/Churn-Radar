"""Avaliação e comparação de modelos de churn."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Calcula métricas de classificação para um modelo treinado."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }


def confusion(model, X_test, y_test):
    """Retorna a matriz de confusão (linhas = real, colunas = previsto)."""
    return confusion_matrix(y_test, model.predict(X_test))


def compare_models(trained: dict, X_test, y_test) -> pd.DataFrame:
    """Tabela comparativa das métricas dos modelos, ordenada por PR-AUC."""
    rows = []
    for name, model in trained.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("Model").round(4)
    return df.sort_values("PR_AUC", ascending=False).reset_index()
