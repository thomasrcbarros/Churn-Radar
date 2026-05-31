# Churn-Radar

Modelagem preditiva de churn em telecom. Comparação de 3 modelos de ML com análise de
interpretabilidade e métricas de negócio para estratégia de retenção de clientes.

## Dataset

[Telecom Churn](https://www.kaggle.com/datasets/barun2104/telecom-churn) (`barun2104/telecom-churn`):
~3.333 clientes, 10 features numéricas e o alvo binário `Churn` (~14,5% positivos).
O acesso é feito via `kagglehub`.

## Modelos selecionados

Escolhidos com a skill **/model-selector** (classificação binária, dados pequenos e
desbalanceados, interpretabilidade relevante, prioridade balanceada):

| Modelo | Papel |
|---|---|
| **Logistic Regression** | Baseline interpretável (odds ratios), `class_weight='balanced'` |
| **Random Forest** | Robusto, não-linear, `class_weight='balanced'` |
| **XGBoost** | Teto de performance em dados tabulares, `scale_pos_weight` |

## Estrutura

```
src/
  config.py     # caminhos, semente, parâmetros de negócio (CLV, custos)
  data.py       # download via kagglehub + carga do CSV
  features.py   # split treino/teste estratificado
  train.py      # treina e salva os 3 modelos
  evaluate.py   # ROC-AUC, PR-AUC, precision/recall/F1, matriz de confusão
  interpret.py  # SHAP (RF/XGB) e coeficientes (LogReg)
  business.py   # threshold ótimo e ROI de retenção
notebooks/
  churn_analysis.ipynb   # EDA + SHAP + métricas de negócio
```

## Setup

```bash
pip install -r requirements.txt          # kagglehub fica pinado em 1.0.0
cp .env.example .env                      # preencha KAGGLE_USERNAME e KAGGLE_KEY
```

O arquivo `.env` é ignorado pelo git (ver `.gitignore`) e **nunca** deve ser commitado.

## Como rodar

```bash
python -m src.train      # baixa os dados, treina os 3 modelos e imprime a comparação
python -m src.interpret  # SHAP/coeficientes dos modelos salvos (gera models/shap_*.png)
jupyter notebook notebooks/churn_analysis.ipynb   # EDA, SHAP e ROI
```

> **Nota:** o acesso ao Kaggle costuma estar bloqueado em ambientes de nuvem. O download
> dos dados e o treino completo devem ser executados **localmente** com um `.env` válido.
