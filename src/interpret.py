"""Interpretabilidade dos modelos de churn.

Centraliza a lógica de SHAP (modelos de árvore) e de coeficientes (Logistic
Regression), para que possa ser usada tanto pelo notebook quanto via script
(`python -m src.interpret`). Salva os gráficos em `models/` por padrão.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from . import config


def tree_shap_values(model, X):
    """Retorna (explainer, shap_values) para um modelo de árvore (RF/XGB)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Para classificadores binários os explainers podem retornar a classe
    # positiva de duas formas: uma lista [classe_0, classe_1] (SHAP antigo)
    # ou um array 3D (n, features, classes) (SHAP novo, ex. RandomForest).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]
    return explainer, shap_values


def shap_summary(model, X, save_path: Path | None = None):
    """Gera o summary plot SHAP de um modelo de árvore."""
    _, shap_values = tree_shap_values(model, X)
    shap.summary_plot(shap_values, X, show=save_path is None)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close()
    return save_path


def mean_abs_shap(model, X) -> pd.Series:
    """Importância global = média do |SHAP| por feature (ordem decrescente)."""
    _, shap_values = tree_shap_values(model, X)
    importance = pd.Series(
        abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    return importance


def logreg_coefficients(pipeline, feature_names) -> pd.Series:
    """Coeficientes da Logistic Regression (dentro de um Pipeline com scaler)."""
    clf = pipeline.named_steps["clf"]
    return pd.Series(clf.coef_[0], index=feature_names).sort_values()


def explain_all(models: dict, X) -> dict:
    """Calcula interpretabilidade para todos os modelos.

    Retorna um dict por modelo: importância (SHAP) para árvores e coeficientes
    para a Logistic Regression.
    """
    out = {}
    for name, model in models.items():
        if name == "LogisticRegression":
            out[name] = logreg_coefficients(model, X.columns)
        else:
            out[name] = mean_abs_shap(model, X)
    return out


if __name__ == "__main__":
    # Smoke run com os modelos treinados e salvos por src.train.
    import matplotlib

    matplotlib.use("Agg")  # backend sem display, para rodar via script/CI
    import joblib

    from .data import load_data
    from .features import make_splits

    df = load_data()
    _, X_test, _, _ = make_splits(df)

    for name in ("RandomForest", "XGBoost"):
        path = config.MODELS_DIR / f"{name}.joblib"
        if path.exists():
            model = joblib.load(path)
            print(f"\n=== Importância SHAP (média |SHAP|) — {name} ===")
            print(mean_abs_shap(model, X_test).round(4).to_string())
            shap_summary(model, X_test, config.MODELS_DIR / f"shap_{name}.png")
            print(f"summary plot salvo em models/shap_{name}.png")

    logreg_path = config.MODELS_DIR / "LogisticRegression.joblib"
    if logreg_path.exists():
        logreg = joblib.load(logreg_path)
        print("\n=== Coeficientes — LogisticRegression ===")
        print(logreg_coefficients(logreg, X_test.columns).round(4).to_string())
