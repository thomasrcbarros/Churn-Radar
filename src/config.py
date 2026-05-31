"""Configuração central do projeto: caminhos, semente e parâmetros de negócio."""

from pathlib import Path

# --- Caminhos ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# --- Dataset Kaggle ---
KAGGLE_DATASET = "barun2104/telecom-churn"
TARGET_COL = "Churn"

# --- Reprodutibilidade ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Hiperparâmetros tunados (RandomizedSearchCV, otimizando PR-AUC) ---
# Obtidos via `python -m src.tune`. Fixados aqui para reprodutibilidade.
TUNED_PARAMS = {
    "LogisticRegression": {
        "C": 0.0019,
        "penalty": "l2",
        "solver": "liblinear",
    },
    "RandomForest": {
        "n_estimators": 456,
        "max_depth": 5,
        "min_samples_leaf": 12,
        "max_features": 0.5,
    },
    "XGBoost": {
        "n_estimators": 392,
        "max_depth": 3,
        "learning_rate": 0.0304,
        "subsample": 0.9549,
        "colsample_bytree": 0.7798,
        "min_child_weight": 8,
        "gamma": 0.2862,
        "reg_alpha": 0.4998,
        "reg_lambda": 1.8841,
    },
}

# --- Parâmetros de negócio (para o cálculo de ROI de retenção) ---
# Valores ilustrativos; ajuste conforme a realidade do negócio.
CLV = 1000.0              # Customer Lifetime Value de um cliente retido (R$)
RETENTION_COST = 100.0    # Custo de uma ação/campanha de retenção por cliente (R$)
RETENTION_SUCCESS = 0.30  # Probabilidade de a ação reverter o churn (30%)

# Garante que os diretórios de saída existam.
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
