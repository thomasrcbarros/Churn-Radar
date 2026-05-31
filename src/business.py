"""Métricas de negócio: threshold ótimo e ROI de estratégia de retenção.

Modelo econômico simplificado por cliente classificado como "vai dar churn"
(positivo previsto), sobre o qual uma ação de retenção é tomada:

  - Verdadeiro positivo (TP): cliente realmente daria churn. Com probabilidade
    RETENTION_SUCCESS a ação o retém → ganho = CLV * RETENTION_SUCCESS.
    Custo da ação = RETENTION_COST.
  - Falso positivo (FP): cliente não daria churn, mas gastamos a ação à toa.
    Custo = RETENTION_COST.
  - Negativos previstos: nenhuma ação, sem custo (mas TP perdido se errarmos).

Lucro = soma sobre clientes-alvo de (ganho esperado - custo da ação).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def profit_at_threshold(y_true, y_proba, threshold: float) -> float:
    """Lucro líquido esperado da campanha de retenção dado um threshold."""
    targeted = y_proba >= threshold
    tp = int(np.sum(targeted & (y_true == 1)))
    fp = int(np.sum(targeted & (y_true == 0)))

    gain = tp * config.CLV * config.RETENTION_SUCCESS
    cost = (tp + fp) * config.RETENTION_COST
    return gain - cost


def optimize_threshold(y_true, y_proba, steps: int = 101) -> pd.DataFrame:
    """Varre thresholds e retorna lucro por threshold (ordenado por lucro)."""
    y_true = np.asarray(y_true)
    thresholds = np.linspace(0.0, 1.0, steps)
    rows = [
        {"threshold": round(t, 3), "profit": profit_at_threshold(y_true, y_proba, t)}
        for t in thresholds
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("profit", ascending=False).reset_index(drop=True)


def baseline_profit(y_true) -> float:
    """Cenário 'sem modelo': agir sobre todos os clientes (threshold = 0)."""
    y_true = np.asarray(y_true)
    return profit_at_threshold(y_true, np.ones(len(y_true)), 0.0)


def roi_report(y_true, y_proba) -> dict:
    """Compara lucro ótimo do modelo vs. baseline (agir sobre todos)."""
    grid = optimize_threshold(y_true, y_proba)
    best = grid.iloc[0]
    base = baseline_profit(y_true)
    n_targeted = int(np.sum(np.asarray(y_proba) >= best["threshold"]))
    return {
        "best_threshold": float(best["threshold"]),
        "model_profit": float(best["profit"]),
        "baseline_profit": float(base),
        "uplift": float(best["profit"] - base),
        "n_targeted": n_targeted,
    }


def save_profit_curve(y_true, y_proba, best_threshold, save_path) -> "object":
    """Salva o gráfico de lucro esperado vs. threshold, marcando o ótimo."""
    import matplotlib

    matplotlib.use("Agg")  # backend sem display, para rodar via script/CI
    import matplotlib.pyplot as plt

    grid = optimize_threshold(y_true, y_proba).sort_values("threshold")
    plt.figure(figsize=(8, 5))
    plt.plot(grid["threshold"], grid["profit"])
    plt.axvline(best_threshold, color="r", ls="--", label="threshold ótimo")
    plt.axhline(0, color="grey", lw=0.8)
    plt.xlabel("threshold")
    plt.ylabel("lucro esperado (R$)")
    plt.title("Lucro da campanha de retenção vs. threshold")
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    return save_path


if __name__ == "__main__":
    # Calcula o ROI dos modelos treinados/salvos por src.train.
    import joblib

    from .data import load_data
    from .features import make_splits

    df = load_data()
    _, X_test, _, y_test = make_splits(df)
    y_true = y_test.values

    print("Parâmetros de negócio: "
          f"CLV={config.CLV}, custo={config.RETENTION_COST}, "
          f"sucesso={config.RETENTION_SUCCESS:.0%}")
    rows = []
    for name in ("LogisticRegression", "RandomForest", "XGBoost"):
        path = config.MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        model = joblib.load(path)
        proba = model.predict_proba(X_test)[:, 1]
        rep = roi_report(y_true, proba)
        rep["Model"] = name
        rows.append(rep)
        save_profit_curve(
            y_true, proba, rep["best_threshold"],
            config.MODELS_DIR / f"profit_{name}.png",
        )

    if rows:
        cols = ["Model", "best_threshold", "n_targeted",
                "model_profit", "baseline_profit", "uplift"]
        table = pd.DataFrame(rows)[cols].sort_values("model_profit", ascending=False)
        print("\n=== ROI da estratégia de retenção (teste) ===")
        print(table.to_string(index=False))
        print("\nCurvas de lucro salvas em models/profit_<modelo>.png")
    else:
        print("Nenhum modelo salvo encontrado. Rode `python -m src.train` antes.")
