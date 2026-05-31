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

# --- Parâmetros de negócio (para o cálculo de ROI de retenção) ---
# Valores ilustrativos; ajuste conforme a realidade do negócio.
CLV = 1000.0              # Customer Lifetime Value de um cliente retido (R$)
RETENTION_COST = 100.0    # Custo de uma ação/campanha de retenção por cliente (R$)
RETENTION_SUCCESS = 0.30  # Probabilidade de a ação reverter o churn (30%)

# Garante que os diretórios de saída existam.
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
