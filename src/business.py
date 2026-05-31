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
    return {
        "best_threshold": float(best["threshold"]),
        "model_profit": float(best["profit"]),
        "baseline_profit": float(base),
        "uplift": float(best["profit"] - base),
    }
